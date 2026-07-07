Vinf(X, t) = magVinf * [cosd(AOA), 0.0, sind(AOA)]

wing = make_cantilever_template(
    span, root_chord, tip_chord, leading_edge_sweep, 0.0, twist_root, twist_tip, n_span
)
wing_ref = deepcopy(wing)

system = vlm.WingSystem()
vlm.addwing(system, "Wing", wing)
vehicle = uns.VLMVehicle(system; vlm_system=system, wake_system=system)

Vvehicle(t) = zeros(3)
anglevehicle(t) = zeros(3)
maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

simulation = uns.Simulation(
    vehicle, maneuver, 0.0, 0.0, ttot;
    Vinit=zeros(3), Winit=zeros(3)
)


repo_root = normpath(joinpath(@__DIR__, ".."))
save_path = normpath(joinpath(repo_root, "results", "fluid"))
run_name = "fluid"
mkpath(save_path)


max_particles = Int((nsteps + 1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step + 1) + p_per_step))
omit_shedding_rows = Int[]

rmv_strength = 2 * 2 / max(p_per_step, 1) * dt / (1 / 12)
minmaxGamma = rmv_strength .* [wake_strength_min_factor, wake_strength_max_factor]
wake_treatment_strength = uns.remove_particles_strength(
    minmaxGamma[1]^2, minmaxGamma[2]^2; every_nsteps=max(wake_remove_every, 1)
)

minmaxsigma = sigma_vpm_overwrite .* [wake_sigma_min_factor, wake_sigma_max_factor]
wake_treatment_sigma = uns.remove_particles_sigma(
    minmaxsigma[1], minmaxsigma[2]; every_nsteps=max(wake_remove_every, 1)
)

wake_treatment_sphere = uns.remove_particles_sphere(
    (wake_sphere_factor * b)^2, 1; Xoff=[0.5 * b, 0.0, 0.0]
)

wake_treatment = uns.concatenate(
    wake_treatment_sphere,
    wake_treatment_strength,
    wake_treatment_sigma
)

vpm_fmm_settings = vpm.FMM(
    p=4,
    ncrit=50,
    theta=0.4,
    shrink_recenter=true,
    relative_tolerance=1e-3,
    absolute_tolerance=1e-3,
    autotune_p=true,
    autotune_ncrit=true,
    autotune_reg_error=false,
    default_rho_over_sigma=1.0,
    min_ncrit=3
)

vpm_viscous = vpm.CoreSpreading(nu, sigma_vpm_overwrite, 1.0)

# Check if it is correct?-----------------------------------------------------
sock = connect_to_server(cfg.coupling.host, cfg.coupling.port)

m_span = vlm.get_m(wing)
ys_ref = [wing_ref._ym[i] for i in 1:m_span]
eta_span_fluid = [clamp(ys_ref[i] / span, 0.0, 1.0) for i in 1:m_span]
eta_span_coupling = build_coupling_eta_span(m_span, solid_ny_for_sampling, span_sampling_mode)
if span_sampling_mode == "custom-stride"
    assert_eta_close("Fluid VLM eta vs custom coupling eta", eta_span_fluid, eta_span_coupling)
end
eta_span_force_payload = span_sampling_mode == "custom-stride" ? eta_span_coupling : eta_span_fluid

# For single-row spanwise wing, communication force channel is one chord location.
eta_chord_force = [eta_cp]

u_prev_cp = zeros(Float64, m_span, 3)
omega_prev_cp = zeros(Float64, m_span, 3)
forces_prev = zeros(Float64, m_span, 3)

r_vortex_ref = zeros(Float64, m_span, 3)
r_le_ref = zeros(Float64, m_span, 3)
r_te_ref = zeros(Float64, m_span, 3)
for i in 1:m_span
    cp = [wing_ref._xm[i], wing_ref._ym[i], wing_ref._zm[i]]
    r_vortex_ref[i, :] .= [wing_ref._xn[i] - cp[1], wing_ref._yn[i] - cp[2], wing_ref._zn[i] - cp[3]]
    r_le_ref[i, :] .= [wing_ref._xlwingdcr[i] - cp[1], wing_ref._ywingdcr[i] - cp[2], wing_ref._zlwingdcr[i] - cp[3]]
    r_te_ref[i, :] .= [wing_ref._xtwingdcr[i] - cp[1], wing_ref._ywingdcr[i] - cp[2], wing_ref._ztwingdcr[i] - cp[3]]
end

fluid_geom_csv = joinpath(save_path, run_name * "_fluid_cp_le_te_coords.csv")
open(fluid_geom_csv, "w") do io
    println(io, "index,i_span,x_cp,y_cp,z_cp,x_le,y_le,z_le,x_te,y_te,z_te")
    for i in 1:m_span
        println(
            io,
            string(
                i, ",", i, ",",
                wing_ref._xm[i], ",", wing_ref._ym[i], ",", wing_ref._zm[i], ",",
                wing_ref._xlwingdcr[i], ",", wing_ref._ywingdcr[i], ",", wing_ref._zlwingdcr[i], ",",
                wing_ref._xtwingdcr[i], ",", wing_ref._ywingdcr[i], ",", wing_ref._ztwingdcr[i]
            )
        )
    end
end

if DEBUG_IO
    println("Coordinate system check: x=chord, y=span, z=normal")
    println("FLUID first CP/LE/TE = ", [wing_ref._xm[1], wing_ref._ym[1], wing_ref._zm[1]], " / ",
            [wing_ref._xlwingdcr[1], wing_ref._ywingdcr[1], wing_ref._zlwingdcr[1]], " / ",
            [wing_ref._xtwingdcr[1], wing_ref._ywingdcr[1], wing_ref._ztwingdcr[1]])
    println("Saved fluid geometry coordinates: ", fluid_geom_csv)
end

step_hist = Int[]
force_res_hist = Float64[]
geom_res_hist = Float64[]
force_trace_path = joinpath(save_path, run_name * "_force_payload_history.jsonl")
force_trace_io = open(force_trace_path, "w")

diag_path = joinpath(save_path, run_name * "_coupling_diagnostics.csv")


# Initial geometry
msg0 = read_json_line(sock, "init")
if haskey(msg0, "dt")
    dt_solid = Float64(msg0["dt"])
    rel = abs(dt_solid - dt) / max(abs(dt), 1.0e-16)
    rel > 1.0e-10 && @warn "Solid/fluid dt mismatch at init: solid=$(dt_solid), fluid=$(dt)"
end
if haskey(msg0, "n_span")
    nspan_solid = Int(msg0["n_span"])
    if nspan_solid != m_span
        error("Spanwise mismatch at init: solid n_span=$(nspan_solid), fluid m_span=$(m_span). " *
              "Set FLUID_N_SPAN or COUPLING_NSPAN_COMM consistently.")
    end
end
if span_sampling_mode == "custom-stride" && haskey(msg0, "eta_span") && length(msg0["eta_span"]) == m_span
    eta_span_solid = Float64.(msg0["eta_span"])
    assert_eta_close("Solid eta_span vs fluid coupling eta", eta_span_solid, eta_span_coupling)
end
apply_from_message!(msg0; first_step=true, step=0)

step_ref = Ref(0)
use_ftot_force = Ref(true)


runtime_pipeline = uns.concatenate(wake_treatment, coupling_runtime_function)


