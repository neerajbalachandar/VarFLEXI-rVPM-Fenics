# Particle shedding enabled for flow field visualization, chordwise discretization enabled and geometry superimposed, 
# along with better and correct particle shedding, shedding stabilised with flowunsteady sample codesm, work conservation, AOA 20deg

using Sockets
using JSON
using LinearAlgebra
import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm

# Avoid FLOWVLM colinearity edge-case crash when Gamma is `nothing` in
# geometric-factor evaluations.
vlm.VLMSolver._regularize(true)

# Workaround for FLOWVLM colinearity bug:
# when gamma===nothing, promote_type can become Union{Nothing,Float64}, and
# zeros(::Type{Union{Nothing,Float64}}, 3) throws.

#Understand what is done here
function vlm.VLMSolver._V_AB(A::Vector{<:vlm.VLMSolver.FWrap}, B, C, gamma; ign_col::Bool=false)
    r0 = B - A
    r1 = C - A
    r2 = C - B
    crss = LinearAlgebra.cross(r1, r2)
    magsqr = LinearAlgebra.dot(crss, crss) + (vlm.VLMSolver.regularize ? vlm.VLMSolver.core_rad : 0)

    TF = gamma === nothing ? promote_type(eltype(A), eltype(B), eltype(C)) :
                             promote_type(eltype(A), eltype(B), eltype(C), typeof(gamma))

    if vlm.VLMSolver._check_collinear(magsqr / LinearAlgebra.norm(r0), vlm.VLMSolver.col_crit; ign_col=ign_col)
        if ign_col == false && vlm.VLMSolver.n_col == 1 && vlm.VLMSolver.mute_warning == false
            println("\n\t magsqr:$magsqr \n\t A:$A \n\t B:$B \n\t C:$C")
        end
        return zeros(TF, 3)
    end

    F1 = crss / magsqr
    aux = r1 / sqrt(LinearAlgebra.dot(r1, r1)) - r2 / sqrt(LinearAlgebra.dot(r2, r2))
    F2 = LinearAlgebra.dot(r0, aux)

    if vlm.VLMSolver.blobify
        F1 *= vlm.VLMSolver.gw(LinearAlgebra.norm(crss) / LinearAlgebra.norm(r0), vlm.VLMSolver.smoothing_rad)
    end

    return gamma === nothing ? (F1 * F2) : ((gamma / 4 / pi) * F1 * F2)
end

function vlm.VLMSolver._V_Ainf_out(A::Vector{<:vlm.VLMSolver.FWrap},
                                   infD::Vector{<:vlm.VLMSolver.FWrap}, C, gamma;
                                   ign_col::Bool=false)
    AC = C - A
    unitinfD = infD / sqrt(LinearAlgebra.dot(infD, infD))
    AAp = LinearAlgebra.dot(unitinfD, AC) * unitinfD
    Ap = AAp + A

    boundAAp = vlm.VLMSolver._V_AB(A, Ap, C, gamma; ign_col=ign_col)

    ApC = C - Ap
    crss = LinearAlgebra.cross(infD, ApC)
    mag = sqrt(LinearAlgebra.dot(crss, crss) + (vlm.VLMSolver.regularize ? vlm.VLMSolver.core_rad : 0))

    TF = gamma === nothing ? promote_type(eltype(A), eltype(infD), eltype(C)) :
                             promote_type(eltype(A), eltype(infD), eltype(C), typeof(gamma))

    if vlm.VLMSolver._check_collinear(mag, vlm.VLMSolver.col_crit; ign_col=ign_col)
        return zeros(TF, 3)
    end

    h = mag / sqrt(LinearAlgebra.dot(infD, infD))
    n = crss / mag
    F = n / h

    if vlm.VLMSolver.blobify
        F *= vlm.VLMSolver.gw(h, vlm.VLMSolver.smoothing_rad)
    end

    return gamma === nothing ? (F + boundAAp) : ((gamma / 4 / pi) * F + boundAAp)
end











#------------------------------------------------------MAIN----------------------------------------------------
# SIMULATION PARAMETERS
AOA             = 20.0
magVinf         = 10
rho             = 0.10

# Match solid geometry (cantilever wing: y in [0, span])
span            = 1.0
root_chord      = 0.12
tip_chord       = 0.08
leading_edge_sweep = 0.0

b               = span
ar              = span / tip_chord
tr              = tip_chord / root_chord
twist_root      = 0.0
twist_tip       = 0.0
gamma           = 0.0

n_span          = 80

wakelength      = 2.75*b
ttot            = 4.0   # wakelength/magVinf
nsteps          = 400
dt              = ttot/nsteps

# VPM parameters
# FLOWUnsteady expects integer particle release count per step.
p_per_step      = 1
lambda_vpm      = 2.0
sigma_vpm_overwrite = lambda_vpm * magVinf * dt / p_per_step
sigma_vlm_solver = -1
sigma_vlm_surf   = 0.05*b
shed_starting    = false
unsteady_shedcrit = 0.01
vlm_rlx          = 0.35

# Coupling stabilization (numerical damping)
geom_relax       = 0.20           # 0<geom_relax<=1; lower is more damping
force_relax      = 0.20           # 0<force_relax<=1
max_abs_disp     = 0.01*b         # clamp incoming displacement magnitude
max_abs_force    = 1.0e6          # clamp outgoing per-panel force component
max_abs_gamma    = 1.0e4          # cap pathological circulation spikes
disp_scale_x     = 1.00           # debug alignment: apply full displacement
disp_scale_y     = 1.00           # debug alignment: apply full displacement
disp_scale_z     = 1.00           # full normal update

# 2D coupling grid (span x chord) used for socket data exchange
n_chord     = 12
eta_chord_edges  = collect(range(0.0, 1.0; length=n_chord+1))
eta_chord_cp     = [(eta_chord_edges[j] + 0.75*(eta_chord_edges[j+1]-eta_chord_edges[j]))
                     for j in 1:n_chord]
# Communication chord coordinates are the panel control-point locations.
eta_chord_comm   = copy(eta_chord_cp)
eta_chord_vortex = [(eta_chord_edges[j] + 0.25*(eta_chord_edges[j+1]-eta_chord_edges[j]))
                     for j in 1:n_chord]
eta_chord_le     = [eta_chord_edges[j] for j in 1:n_chord]
eta_chord_te     = [eta_chord_edges[j+1] for j in 1:n_chord]

Vinf(X,t) = magVinf*[cosd(AOA), 0.0, sind(AOA)]


# Primary Question - why do we have a geometry definition here? if the solid fenics has a geom use the same, why confuse?
# GEOMETRY
println("Initializing geometry...")

function chord_length_twist(xl, zl, xt, zt)
    dx = xt - xl
    dz = zt - zl
    c = sqrt(dx*dx + dz*dz)
    twist = atan(-dz, dx) * 180 / pi
    return c, twist
end

function make_cantilever_template(span, c_root, c_tip, x_tip, z_tip, twist_root, twist_tip, nspan)
    wing = vlm.Wing(0.0, 0.0, 0.0, c_root, twist_root)
    vlm.addchord(wing, x_tip, span, z_tip, c_tip, twist_tip, nspan; r=1.0)
    return wing
end

function split_wing_chordwise(wing_base, eta_edges::Vector{Float64})
    nrows = length(eta_edges) - 1
    rows = Vector{typeof(wing_base)}(undef, nrows)
    refs = Vector{typeof(wing_base)}(undef, nrows)

    m = vlm.get_m(wing_base)
    nch = length(wing_base._xlwingdcr)
    @assert nch == m + 1

    for j in 1:nrows
        η0 = eta_edges[j]
        η1 = eta_edges[j+1]
        # Build each chordwise row with FLOWVLM constructors to preserve
        # internal consistency used by horseshoe shedding.
        xl1 = wing_base._xlwingdcr[1]
        yl1 = wing_base._ywingdcr[1]
        zl1 = wing_base._zlwingdcr[1]
        xt1 = wing_base._xtwingdcr[1]
        zt1 = wing_base._ztwingdcr[1]
        xle1 = xl1 + η0 * (xt1 - xl1)
        zle1 = zl1 + η0 * (zt1 - zl1)
        xte1 = xl1 + η1 * (xt1 - xl1)
        zte1 = zl1 + η1 * (zt1 - zl1)
        c1, t1 = chord_length_twist(xle1, zle1, xte1, zte1)
        w = vlm.Wing(xle1, yl1, zle1, c1, t1)

        for i in 2:nch
            xl = wing_base._xlwingdcr[i]
            yl = wing_base._ywingdcr[i]
            zl = wing_base._zlwingdcr[i]
            xt = wing_base._xtwingdcr[i]
            zt = wing_base._ztwingdcr[i]
            xle = xl + η0 * (xt - xl)
            zle = zl + η0 * (zt - zl)
            xte = xl + η1 * (xt - xl)
            zte = zl + η1 * (zt - zl)
            c, t = chord_length_twist(xle, zle, xte, zte)
            vlm.addchord(w, xle, yl, zle, c, t, 1; r=1.0)
        end

        @assert vlm.get_m(w) == m
        rows[j] = w
        refs[j] = deepcopy(w)
    end

    return rows, refs
end

wing_template = make_cantilever_template(
    span, root_chord, tip_chord, leading_edge_sweep, 0.0, twist_root, twist_tip, n_span
)
row_wings, row_wing_refs = split_wing_chordwise(wing_template, eta_chord_edges)

system = vlm.WingSystem()
for j in 1:n_chord
    vlm.addwing(system, "WingRow$j", row_wings[j])
end

vehicle = uns.VLMVehicle(system;
                         vlm_system=system,
                         wake_system=system)

# MANEUVER
Vvehicle(t) = zeros(3)
anglevehicle(t) = zeros(3)

maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

Vref = 0.0
RPMref = 0.0
Vinit = zeros(3)
Winit = zeros(3)

simulation = uns.Simulation(vehicle, maneuver, Vref, RPMref, ttot;
                              Vinit=Vinit, Winit=Winit)

# Output configuration
save_path = normpath(joinpath(@__DIR__, "..", "results", "fluid"))
run_name = "fluid_explicit_vpm_v7_conservative"
mkpath(save_path)

# Maximum number of particles (must be Int for FLOWVPM.ParticleField constructor)
max_particles = Int((nsteps+1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step+1) + p_per_step))

# For chordwise-row decomposition, only the aft-most row represents the
# physical trailing edge and should shed wake particles.
omit_shedding_rows = collect(1:max(0, n_chord-1))

# Wake treatment adapted from standard FLOWUnsteady examples to keep the
# particle field bounded during long coupled runs.
rmv_strength = 2 * 2 / p_per_step * dt / (1 / 12)
minmaxGamma = rmv_strength .* [0.0001, 0.15]
wake_treatment_strength = uns.remove_particles_strength(
    minmaxGamma[1]^2, minmaxGamma[2]^2; every_nsteps=5
)

minmaxsigma = sigma_vpm_overwrite .* [0.1, 6.0]
wake_treatment_sigma = uns.remove_particles_sigma(
    minmaxsigma[1], minmaxsigma[2]; every_nsteps=5
)

wake_treatment_sphere = uns.remove_particles_sphere(
    (3.0 * b)^2, 1; Xoff=[0.5 * b, 0.0, 0.0]
)

wake_treatment = uns.concatenate(
    wake_treatment_sphere,
    wake_treatment_strength,
    wake_treatment_sigma
)

# Robust FMM setup: disable rho/sigma auto-root solve that can fail to bracket
# for extreme particle states in long coupled runs.
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

# GEOMETRY UPDATE
function update_geometry_absolute(wing, wing_ref, u_cp, u_vortex, u_le, u_te)

    m = vlm.get_m(wing)

    @assert size(u_cp,1) == m
    @assert size(u_cp,2) == 3
    @assert size(u_vortex,1) == m
    @assert size(u_le,1) == m
    @assert size(u_te,1) == m

    # --- Control points ---
    for i in 1:m
        wing._xm[i] = wing_ref._xm[i] + u_cp[i,1]
        wing._ym[i] = wing_ref._ym[i] + u_cp[i,2]
        wing._zm[i] = wing_ref._zm[i] + u_cp[i,3]
    end

    # --- Bound vortices ---
    # NOTE: bound-vortex points have length m+1, while profiles are defined at m
    # points. Clamp the last point to the last available displacement.
    for i in 1:(m+1)
        idx = min(i, m)
        wing._xn[i] = wing_ref._xn[i] + u_vortex[idx,1]
        wing._yn[i] = wing_ref._yn[i] + u_vortex[idx,2]
        wing._zn[i] = wing_ref._zn[i] + u_vortex[idx,3]
    end

    # --- Leading & trailing edges ---
    nch = length(wing._xlwingdcr)
    for i in 1:nch
        idx = min(i, m)
        wing._xlwingdcr[i] = wing_ref._xlwingdcr[i] + u_le[idx,1]
        wing._zlwingdcr[i] = wing_ref._zlwingdcr[i] + u_le[idx,3]
        wing._xtwingdcr[i] = wing_ref._xtwingdcr[i] + u_te[idx,1]
        wing._ztwingdcr[i] = wing_ref._ztwingdcr[i] + u_te[idx,3]
        wing._ywingdcr[i]  = wing_ref._ywingdcr[i]  + 0.5*(u_le[idx,2] + u_te[idx,2])
    end

    # Rebuild horseshoes next call without clearing wing.sol["Gamma"].
    wing._HSs = nothing
end

function read_json_line(sock::TCPSocket, tag::String)
    line = try
        readline(sock)
    catch err
        if err isa EOFError
            error("$tag: coupling socket closed")
        end
        rethrow(err)
    end
    s = String(line)
    if isempty(strip(s))
        error("$tag: received empty line from coupling")
    end
    return JSON.parse(s)
end

uniform_eta(n::Int) = n <= 1 ? [0.0] : collect(range(0.0, 1.0; length=n))

function interp_profile(eta_src::Vector{Float64}, vals::Matrix{Float64}, eta::Float64)
    n = length(eta_src)
    if n == 1
        return copy(vals[1, :])
    end
    e = clamp(eta, eta_src[1], eta_src[end])
    j = searchsortedlast(eta_src, e)
    if j <= 0
        return copy(vals[1, :])
    elseif j >= n
        return copy(vals[end, :])
    else
        e0 = eta_src[j]
        e1 = eta_src[j+1]
        w = (e - e0) / max(e1 - e0, eps(Float64))
        return (1-w).*vals[j, :] .+ w.*vals[j+1, :]
    end
end

function sample_grid_disp(grid::Array{Float64,3}, eta_s_src::Vector{Float64},
                          eta_c_src::Vector{Float64}, eta_s::Float64, eta_c::Float64)
    ns = size(grid, 1)
    tmp = zeros(Float64, ns, 3)
    for i in 1:ns
        vals = reshape(grid[i, :, :], size(grid,2), 3)
        tmp[i, :] .= interp_profile(eta_c_src, vals, eta_c)
    end
    return interp_profile(eta_s_src, tmp, eta_s)
end

function decode_geometry_payload(msg, eta_span_dst::Vector{Float64}, eta_chord_dst::Vector{Float64})
    if !haskey(msg, "geometry")
        error("Geometry payload missing key \"geometry\"")
    end
    geo_raw = msg["geometry"]
    ng = length(geo_raw)
    if ng == 0
        error("Received empty geometry array from coupling")
    end

    m = length(eta_span_dst)
    nc = length(eta_chord_dst)

    # Parse flat payload as ng x 3
    geo = zeros(Float64, ng, 3)
    for i in 1:ng
        geo[i,1] = Float64(geo_raw[i][1])
        geo[i,2] = Float64(geo_raw[i][2])
        geo[i,3] = Float64(geo_raw[i][3])
    end

    n_span_in = haskey(msg, "n_span") ? Int(msg["n_span"]) : ng
    n_chord_in = haskey(msg, "n_chord") ? Int(msg["n_chord"]) : 1
    use_2d = n_span_in >= 1 && n_chord_in >= 1 && n_span_in*n_chord_in == ng

    if use_2d
        eta_span_src = haskey(msg, "eta_span") && length(msg["eta_span"]) == n_span_in ?
                       Float64.(msg["eta_span"]) : uniform_eta(n_span_in)
        eta_chord_src = haskey(msg, "eta_chord") && length(msg["eta_chord"]) == n_chord_in ?
                        Float64.(msg["eta_chord"]) : uniform_eta(n_chord_in)

        p_s = sortperm(eta_span_src)
        p_c = sortperm(eta_chord_src)
        eta_span_src = eta_span_src[p_s]
        eta_chord_src = eta_chord_src[p_c]

        grid = reshape(geo, n_span_in, n_chord_in, 3)
        grid_sorted = similar(grid)
        for i in 1:n_span_in, j in 1:n_chord_in
            grid_sorted[i, j, :] .= grid[p_s[i], p_c[j], :]
        end

        u_grid = zeros(Float64, m, nc, 3)
        for i in 1:m
            for j in 1:nc
                u_grid[i, j, :] .= sample_grid_disp(
                    grid_sorted, eta_span_src, eta_chord_src, eta_span_dst[i], eta_chord_dst[j]
                )
            end
        end
        return u_grid, ng, true
    else
        # Backward-compatible span-only mapping.
        u_span = zeros(Float64, m, 3)
        if ng == m
            u_span .= geo
        elseif ng == 1
            for i in 1:m
                u_span[i, :] .= geo[1, :]
            end
        else
            @warn "Geometry panel count mismatch (solid=$ng, fluid=$m). Resampling spanwise."
            for i in 1:m
                s = 1 + (i-1) * (ng-1) / max(m-1, 1)
                i0 = floor(Int, s)
                i1 = ceil(Int, s)
                w = s - i0
                u_span[i,:] .= (1-w).*geo[i0,:] .+ w.*geo[i1,:]
            end
        end
        u_grid = zeros(Float64, m, nc, 3)
        for j in 1:nc
            u_grid[:, j, :] .= u_span
        end
        return u_grid, ng, false
    end
end

function sample_chordwise_fields(u_grid::Array{Float64,3},
                                 eta_src::Vector{Float64},
                                 eta_queries::Vector{Float64})
    m = size(u_grid, 1)
    nq = length(eta_queries)
    out = zeros(Float64, m, nq, 3)
    for i in 1:m
        vals = reshape(u_grid[i, :, :], length(eta_src), 3)
        for j in 1:nq
            out[i, j, :] .= interp_profile(eta_src, vals, eta_queries[j])
        end
    end
    return out
end



# SOCKET CONNECTION

println("Connecting to coupling server...")
sock = connect("127.0.0.1", 9000)
println("Fluid connected.")
write(sock, JSON.json(Dict("role"=>"fluid")) * "\n")
flush(sock)

m_span = vlm.get_m(row_wings[1])
ys_ref = [row_wing_refs[1]._ym[i] for i in 1:m_span]
eta_span_fluid = [clamp(ys_ref[i] / span, 0.0, 1.0) for i in 1:m_span]
u_prev = zeros(Float64, m_span, n_chord, 3)
forces_prev = zeros(Float64, m_span, n_chord, 3)

# Receive initial geometry from solid before launching the continuous run.
msg0 = read_json_line(sock, "init")
u_raw0, _, used2d0 = decode_geometry_payload(msg0, eta_span_fluid, eta_chord_comm)
u_raw0[:, :, 1] .*= disp_scale_x
u_raw0[:, :, 2] .*= disp_scale_y
u_raw0[:, :, 3] .*= disp_scale_z
u_raw0 .= clamp.(u_raw0, -max_abs_disp, max_abs_disp)
u0 = geom_relax .* u_raw0 .+ (1 - geom_relax) .* u_prev
u_prev .= u0
u_vortex0 = sample_chordwise_fields(u0, eta_chord_comm, eta_chord_vortex)
u_le0 = sample_chordwise_fields(u0, eta_chord_comm, eta_chord_le)
u_te0 = sample_chordwise_fields(u0, eta_chord_comm, eta_chord_te)
for j in 1:n_chord
    update_geometry_absolute(
        row_wings[j], row_wing_refs[j],
        u0[:, j, :], u_vortex0[:, j, :], u_le0[:, j, :], u_te0[:, j, :]
    )
    if !haskey(row_wings[j].sol, "Gamma")
        row_wings[j].sol["Gamma"] = zeros(m_span)
    end
end
if used2d0
    println("INFO: initial geometry mapped using 2D span/chord payload")
end

step_ref = Ref(0)
use_ftot_force = Ref(false)
function ensure_gamma!(wing, m)
    if !haskey(wing.sol, "Gamma") || length(wing.sol["Gamma"]) != m
        wing.sol["Gamma"] = zeros(m)
    end
end

function coupling_runtime_function(sim, PFIELD, T, DT; vprintln=(s)->nothing)
    step_ref[] += 1
    step = step_ref[]
    m = m_span

    # Extract panel forces directly from each chordwise row.
    # Use FLOWVLM force postprocessing when available (per-panel Ftot), with
    # Gamma-based fallback for robustness.
    force2d = Vector{Vector{Float64}}(undef, m * n_chord)
    for j in 1:n_chord
        ensure_gamma!(row_wings[j], m)
        Γj = row_wings[j].sol["Gamma"]
        for i in eachindex(Γj)
            γ = Γj[i]
            if !isfinite(γ)
                γ = 0.0
            end
            Γj[i] = clamp(γ, -max_abs_gamma, max_abs_gamma)
        end
        frow = nothing
        if use_ftot_force[]
            try
                vlm.calculate_field(row_wings[j], "Ftot"; rhoinf=rho, t=T)
                if haskey(row_wings[j].sol, "Ftot")
                    frow = row_wings[j].sol["Ftot"]
                end
            catch err
                # Disable repeated failing calls and keep stable fallback.
                use_ftot_force[] = false
                @warn "Disabling Ftot-based panel force extraction; falling back to Gamma-based force. Root cause: $(sprint(showerror, err))"
            end
        end
        for i in 1:m
            Γ = Γj[i]
            if !isfinite(Γ)
                Γ = 0.0
            end
            Γ = clamp(Γ, -max_abs_gamma, max_abs_gamma)

            fx_raw, fy_raw, fz_raw = 0.0, 0.0, 0.0
            if frow != nothing && i <= length(frow)
                fi = frow[i]
                if length(fi) == 3 && all(isfinite, fi)
                    fx_raw = clamp(Float64(fi[1]), -max_abs_force, max_abs_force)
                    fy_raw = clamp(Float64(fi[2]), -max_abs_force, max_abs_force)
                    fz_raw = clamp(Float64(fi[3]), -max_abs_force, max_abs_force)
                else
                    Xcp = [row_wings[j]._xm[i], row_wings[j]._ym[i], row_wings[j]._zm[i]]
                    Vloc = Vinf(Xcp, T)
                    lvec = [
                        row_wings[j]._xn[i+1] - row_wings[j]._xn[i],
                        row_wings[j]._yn[i+1] - row_wings[j]._yn[i],
                        row_wings[j]._zn[i+1] - row_wings[j]._zn[i],
                    ]
                    Fkj = rho * Γ * cross(Vloc, lvec)
                    fx_raw = clamp(Fkj[1], -max_abs_force, max_abs_force)
                    fy_raw = clamp(Fkj[2], -max_abs_force, max_abs_force)
                    fz_raw = clamp(Fkj[3], -max_abs_force, max_abs_force)
                end
            else
                Xcp = [row_wings[j]._xm[i], row_wings[j]._ym[i], row_wings[j]._zm[i]]
                Vloc = Vinf(Xcp, T)
                lvec = [
                    row_wings[j]._xn[i+1] - row_wings[j]._xn[i],
                    row_wings[j]._yn[i+1] - row_wings[j]._yn[i],
                    row_wings[j]._zn[i+1] - row_wings[j]._zn[i],
                ]
                Fkj = rho * Γ * cross(Vloc, lvec)
                fx_raw = clamp(Fkj[1], -max_abs_force, max_abs_force)
                fy_raw = clamp(Fkj[2], -max_abs_force, max_abs_force)
                fz_raw = clamp(Fkj[3], -max_abs_force, max_abs_force)
            end

            fx = force_relax * fx_raw + (1 - force_relax) * forces_prev[i, j, 1]
            fy = force_relax * fy_raw + (1 - force_relax) * forces_prev[i, j, 2]
            fz = force_relax * fz_raw + (1 - force_relax) * forces_prev[i, j, 3]

            forces_prev[i, j, 1] = fx
            forces_prev[i, j, 2] = fy
            forces_prev[i, j, 3] = fz
            idx = (i - 1) * n_chord + j
            force2d[idx] = [fx, fy, fz]
        end
    end
    println("DEBUG(step=$step): first force = ", force2d[1])

    write(sock, JSON.json(Dict(
        "step"=>step,
        "n_span"=>m,
        "n_chord"=>n_chord,
        "eta_span"=>eta_span_fluid,
        "eta_chord"=>eta_chord_comm,
        "force"=>force2d
    ))*"\n")
    flush(sock)

    # Receive geometry for next step and update.
    if step < nsteps
        msg = read_json_line(sock, "step $step")
        u_raw, _, used2d = decode_geometry_payload(msg, eta_span_fluid, eta_chord_comm)
        u_raw[:, :, 1] .*= disp_scale_x
        u_raw[:, :, 2] .*= disp_scale_y
        u_raw[:, :, 3] .*= disp_scale_z
        u_raw .= clamp.(u_raw, -max_abs_disp, max_abs_disp)
        if any(!isfinite, u_raw)
            @warn "Non-finite displacement received at step $step; reusing previous geometry"
            u_raw .= u_prev
        end
        u = geom_relax .* u_raw .+ (1 - geom_relax) .* u_prev
        u_prev .= u
        u_vortex = sample_chordwise_fields(u, eta_chord_comm, eta_chord_vortex)
        u_le = sample_chordwise_fields(u, eta_chord_comm, eta_chord_le)
        u_te = sample_chordwise_fields(u, eta_chord_comm, eta_chord_te)
        for j in 1:n_chord
            update_geometry_absolute(
                row_wings[j], row_wing_refs[j],
                u[:, j, :], u_vortex[:, j, :], u_le[:, j, :], u_te[:, j, :]
            )
            ensure_gamma!(row_wings[j], m)
        end
        if used2d && (step == 1 || step % 20 == 0)
            println("INFO: mapped 2D geometry at step $step")
        end
    end

    # Stop exactly after nsteps coupling exchanges.
    return step >= nsteps
end

runtime_pipeline = uns.concatenate(wake_treatment, coupling_runtime_function)

# Continuous run so wake particles are shed/convected across all time steps.
uns.run_simulation(simulation, nsteps;
    Vinf=Vinf,
    rho=rho,
    p_per_step=Int(p_per_step),
    max_particles=Int(max_particles),
    sigma_vlm_solver=sigma_vlm_solver,
    sigma_vlm_surf=sigma_vlm_surf,
    sigma_rotor_surf=sigma_vlm_surf,
    sigma_vpm_overwrite=sigma_vpm_overwrite,
    vpm_fmm=vpm_fmm_settings,
    shed_starting=shed_starting,
    shed_unsteady=true,
    unsteady_shedcrit=unsteady_shedcrit,
    omit_shedding=omit_shedding_rows,
    wake_coupled=true,
    vlm_rlx=vlm_rlx,
    extra_runtime_function=runtime_pipeline,
    save_path=save_path,
    run_name=run_name,
    create_savepath=false,
    prompt=false,
    nsteps_save=5,
    save_horseshoes=false
)

close(sock)
println("Fluid solver finished.")
println("Fluid outputs saved in: $save_path")
