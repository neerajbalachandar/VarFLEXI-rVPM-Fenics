using Sockets
using JSON
using LinearAlgebra
import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm

# Avoid FLOWVLM colinearity edge-case crash when Gamma is `nothing` in geometric-factor evaluations.
vlm.VLMSolver._regularize(true)

# Workaround for FLOWVLM colinearity bug:
# when gamma===nothing, promote_type can become Union{Nothing,Float64}, and
# zeros(::Type{Union{Nothing,Float64}}, 3) throws.

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
AOA             = parse(Float64, get(ENV, "FLUID_AOA_DEG", "8.0"))
magVinf         = parse(Float64, get(ENV, "FLUID_VINF", "8.0"))
rho             = parse(Float64, get(ENV, "FLUID_RHO", "1.0"))
DEBUG_IO        = lowercase(get(ENV, "COUPLING_DEBUG_IO", "0")) ∉ ("0", "false", "no")

# Match solid geometry (cantilever wing: y in [0, span])
span            = parse(Float64, get(ENV, "WING_SPAN", "0.8"))
root_chord      = parse(Float64, get(ENV, "WING_ROOT_CHORD", "0.12"))
tip_chord       = parse(Float64, get(ENV, "WING_TIP_CHORD", "0.12"))
leading_edge_sweep = parse(Float64, get(ENV, "WING_X_TIP", "0.0"))

b               = span
ar              = span / tip_chord
tr              = tip_chord / root_chord
twist_root      = 0.0
twist_tip       = 0.0
gamma           = 0.0

comm_n_span     = parse(Int, get(ENV, "COUPLING_NSPAN_COMM", "80"))
n_span          = parse(Int, get(ENV, "FLUID_N_SPAN", string(comm_n_span)))
span_sampling_mode = lowercase(strip(get(ENV, "COUPLING_SPAN_SAMPLING", "node-stride")))
solid_ny_for_sampling = parse(Int, get(ENV, "SOLID_NY", "240"))
custom_span_stride_raw = get(ENV, "COUPLING_SPAN_STRIDE", "")

ttot            = parse(Float64, get(ENV, "COUPLING_TTOT", "5.0"))
nsteps          = parse(Int, get(ENV, "COUPLING_NSTEPS", "1000"))
dt              = ttot/nsteps

# VPM parameters
p_per_step      = parse(Int, get(ENV, "FLUID_P_PER_STEP", "2"))
lambda_vpm      = 2.0
sigma_vpm_overwrite = lambda_vpm * magVinf * dt / p_per_step
sigma_vlm_solver = -1
sigma_vlm_surf   = 0.05*b
shed_starting    = true
unsteady_shedcrit = 0.0
use_unsteady_shedding = true
vlm_rlx          = 0.35

println(
    "Fluid run config: AoA=$(AOA) deg, U=$(magVinf) m/s, T=$(ttot) s, " *
    "nsteps=$(nsteps), dt=$(dt), p_per_step=$(p_per_step)"
)

# Coupling stabilization (numerical damping)
geom_relax       = parse(Float64, get(ENV, "FLUID_GEOM_RELAX", "1.0"))
force_relax      = parse(Float64, get(ENV, "FLUID_FORCE_RELAX", "1.0"))

# max_abs_disp     = b         # clamp incoming displacement magnitude

# Run the simulation once without capping
# max_abs_force    = 1.0e6          # clamp outgoing per-panel force component
# max_abs_gamma    = 1.0e4          # cap pathological circulation spikes

disp_scale_x     = parse(Float64, get(ENV, "FLUID_DISP_SCALE_X", "1.0"))
disp_scale_y     = parse(Float64, get(ENV, "FLUID_DISP_SCALE_Y", "1.0"))
disp_scale_z     = parse(Float64, get(ENV, "FLUID_DISP_SCALE_Z", "1.0"))

# 2D coupling grid (span x chord) used for socket data exchange
n_chord     = parse(Int, get(ENV, "FLUID_N_CHORD", get(ENV, "COUPLING_NCHORD_COMM", "8")))
eta_chord_edges  = collect(range(0.0, 1.0; length=n_chord+1))
eta_chord_cp     = [(eta_chord_edges[j] + 0.75*(eta_chord_edges[j+1]-eta_chord_edges[j]))
                     for j in 1:n_chord]
# Communication chord coordinates are the panel control-point locations.
eta_chord_comm   = copy(eta_chord_cp)
eta_chord_vortex = [(eta_chord_edges[j] + 0.25*(eta_chord_edges[j+1]-eta_chord_edges[j]))
                     for j in 1:n_chord]
eta_chord_le     = [eta_chord_edges[j] for j in 1:n_chord]
eta_chord_te     = [eta_chord_edges[j+1] for j in 1:n_chord]

# Optional rotational coupling stabilization.
# max_abs_rotation = 1.0

min_panel_chord = (tip_chord > 0 ? tip_chord : root_chord) / max(n_chord, 1)
convective_cfl = magVinf * dt / max(min_panel_chord, 1.0e-8)
println(
    "Fluid time-scale diagnostic: dt=$(dt), min_panel_chord=$(min_panel_chord), " *
    "U*dt/dx=$(round(convective_cfl, digits=3))"
)
if convective_cfl > 1.0
    @warn "Convective CFL > 1.0. Wake advection can look stripy/under-resolved. Increase nsteps or reduce U."
end

function build_coupling_eta_span(nspan::Int, ny_solid::Int, mode::String)
    nspan < 1 && error("COUPLING_NSPAN_COMM/FLUID_N_SPAN must be >= 1")
    nspan == 1 && return [0.0]
    if mode == "midpoint"
        return [(i - 0.5) / nspan for i in 1:nspan]
    elseif mode == "node-stride"
        stride = max(1, ny_solid ÷ nspan)
        idx = [(i - 1) * stride for i in 1:nspan]
        if length(unique(idx)) < nspan || maximum(idx) > ny_solid
            idx = round.(Int, range(0, max(ny_solid - 1, 0); length=nspan))
        end
        return [idx_i / max(float(ny_solid), 1.0) for idx_i in idx]
    elseif mode == "custom-stride"
        isempty(custom_span_stride_raw) &&
            error("COUPLING_SPAN_SAMPLING=custom-stride requires COUPLING_SPAN_STRIDE")
        stride = parse(Int, custom_span_stride_raw)
        stride < 1 && error("COUPLING_SPAN_STRIDE must be >= 1")
        last_idx = (nspan - 1) * stride
        last_idx > ny_solid && error(
            "Unsafe custom span stride: " *
            "(COUPLING_NSPAN_COMM-1)*COUPLING_SPAN_STRIDE = $(last_idx) " *
            "exceeds SOLID_NY = $(ny_solid). Reduce stride/stations or increase SOLID_NY."
        )
        return [((i - 1) * stride) / max(float(ny_solid), 1.0) for i in 1:nspan]
    elseif mode == "linspace"
        return [(i - 1) / nspan for i in 1:nspan]
    end
    error("Unsupported COUPLING_SPAN_SAMPLING=$(mode)")
end

function assert_eta_close(name::String, a::Vector{Float64}, b::Vector{Float64}; tol=1.0e-10)
    length(a) == length(b) || error("$(name) length mismatch: $(length(a)) vs $(length(b))")
    err = maximum(abs.(a .- b))
    err <= tol || error(
        "$(name) mismatch for COUPLING_SPAN_SAMPLING=$(span_sampling_mode): " *
        "max |eta_a-eta_b| = $(err). The fluid VLM station locations must match " *
        "the coupling eta grid."
    )
end

# If needed switch to tilting the geometry - based on solid solver
Vinf(X,t) = magVinf*[cosd(AOA), 0.0, sind(AOA)]

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

# chordwise discretization
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
row_te_x = [maximum(w._xtwingdcr) for w in row_wing_refs]
println(
    "Row trailing-edge x positions: min=$(minimum(row_te_x)) max=$(maximum(row_te_x)); " *
    "count=$(length(row_te_x))"
)
println(
    "Expected first-particle offset from each shedding line ~ U*dt/(2*p_per_step)=" *
    "$(magVinf*dt/(2*max(p_per_step,1)))"
)

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
save_path = normpath(joinpath(@__DIR__, "..", "results", "fluid", "v11"))
run_name = "fluid_v11"
mkpath(save_path)

# Shedding of vortex particles and dissipation is periodic. why?

# Maximum number of particles (must be Int for FLOWVPM.ParticleField constructor)
max_particles = Int((nsteps+1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step+1) + p_per_step))

# Shedding rows control:
# - default keeps all chordwise rows active
# - set FLUID_SHED_TE_ONLY=1 to shed trailing-edge row only.
shed_te_only = lowercase(get(ENV, "FLUID_SHED_TE_ONLY", "0")) ∉ ("0", "false", "no")
omit_shedding_rows = shed_te_only ? collect(1:max(0, n_chord - 1)) : Int[]
println(
    "v11 shedding rows active: $(n_chord - length(omit_shedding_rows)) / $(n_chord) " *
    "(te_only=$(shed_te_only))"
)

# Wake treatment adapted from standard FLOWUnsteady examples to keep the
# particle field bounded during long coupled runs.
wake_remove_every = parse(Int, get(ENV, "FLUID_WAKE_REMOVE_EVERY", "5"))
wake_sphere_factor = parse(Float64, get(ENV, "FLUID_WAKE_SPHERE_FACTOR", "3.0"))
wake_strength_min_factor = parse(Float64, get(ENV, "FLUID_WAKE_STRENGTH_MIN_FACTOR", "1.0e-4"))
wake_strength_max_factor = parse(Float64, get(ENV, "FLUID_WAKE_STRENGTH_MAX_FACTOR", "0.15"))
wake_sigma_min_factor = parse(Float64, get(ENV, "FLUID_WAKE_SIGMA_MIN_FACTOR", "0.1"))
wake_sigma_max_factor = parse(Float64, get(ENV, "FLUID_WAKE_SIGMA_MAX_FACTOR", "6.0"))

rmv_strength = 2 * 2 / p_per_step * dt / (1 / 12)
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
    # Is this the reason for the periodic gaps between horseshoes - between different displacement acceptance from solid
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

function decode_vector_payload(msg, key::String,
                               eta_span_dst::Vector{Float64},
                               eta_chord_dst::Vector{Float64};
                               allow_missing::Bool=false)
    if !haskey(msg, key)
        if allow_missing
            m = length(eta_span_dst)
            nc = length(eta_chord_dst)
            return zeros(Float64, m, nc, 3), 0, false
        end
        error("Payload missing key \"$key\"")
    end
    raw = msg[key]
    nraw = length(raw)
    if nraw == 0
        if allow_missing
            m = length(eta_span_dst)
            nc = length(eta_chord_dst)
            return zeros(Float64, m, nc, 3), 0, false
        end
        error("Received empty payload array for key \"$key\"")
    end

    m = length(eta_span_dst)
    nc = length(eta_chord_dst)

    vals = zeros(Float64, nraw, 3)
    for i in 1:nraw
        vals[i, 1] = Float64(raw[i][1])
        vals[i, 2] = Float64(raw[i][2])
        vals[i, 3] = Float64(raw[i][3])
    end

    n_span_in = haskey(msg, "n_span") ? Int(msg["n_span"]) : nraw
    n_chord_in = haskey(msg, "n_chord") ? Int(msg["n_chord"]) : 1
    use_2d = n_span_in >= 1 && n_chord_in >= 1 && n_span_in * n_chord_in == nraw

    if use_2d
        eta_span_src = haskey(msg, "eta_span") && length(msg["eta_span"]) == n_span_in ?
                       Float64.(msg["eta_span"]) : uniform_eta(n_span_in)
        eta_chord_src = haskey(msg, "eta_chord") && length(msg["eta_chord"]) == n_chord_in ?
                        Float64.(msg["eta_chord"]) : uniform_eta(n_chord_in)
        indexing = haskey(msg, "indexing") ? String(msg["indexing"]) : "span-major"

        p_s = sortperm(eta_span_src)
        p_c = sortperm(eta_chord_src)
        eta_span_src = eta_span_src[p_s]
        eta_chord_src = eta_chord_src[p_c]

        # IMPORTANT: Build the 2D payload grid explicitly using the declared
        # flattening rule to avoid Julia reshape memory-order ambiguity.
        grid = zeros(Float64, n_span_in, n_chord_in, 3)
        if indexing == "span-major"
            # idx = (i_span-1)*n_chord + j_chord
            idx = 1
            for i in 1:n_span_in
                for j in 1:n_chord_in
                    grid[i, j, 1] = vals[idx, 1]
                    grid[i, j, 2] = vals[idx, 2]
                    grid[i, j, 3] = vals[idx, 3]
                    idx += 1
                end
            end
        elseif indexing == "chord-major"
            # idx = (j_chord-1)*n_span + i_span
            idx = 1
            for j in 1:n_chord_in
                for i in 1:n_span_in
                    grid[i, j, 1] = vals[idx, 1]
                    grid[i, j, 2] = vals[idx, 2]
                    grid[i, j, 3] = vals[idx, 3]
                    idx += 1
                end
            end
        else
            @warn "Unknown indexing='$indexing' in payload key '$key'; defaulting to span-major."
            idx = 1
            for i in 1:n_span_in
                for j in 1:n_chord_in
                    grid[i, j, 1] = vals[idx, 1]
                    grid[i, j, 2] = vals[idx, 2]
                    grid[i, j, 3] = vals[idx, 3]
                    idx += 1
                end
            end
        end
        grid_sorted = similar(grid)
        for i in 1:n_span_in, j in 1:n_chord_in
            grid_sorted[i, j, :] .= grid[p_s[i], p_c[j], :]
        end

        out = zeros(Float64, m, nc, 3)
        for i in 1:m
            for j in 1:nc
                out[i, j, :] .= sample_grid_disp(
                    grid_sorted, eta_span_src, eta_chord_src, eta_span_dst[i], eta_chord_dst[j]
                )
            end
        end
        return out, nraw, true
    end

    # Backward-compatible span-only mapping.
    out_span = zeros(Float64, m, 3)
    if nraw == m
        out_span .= vals
    elseif nraw == 1
        for i in 1:m
            out_span[i, :] .= vals[1, :]
        end
    else
        @warn "Payload count mismatch for key \"$key\" (recv=$nraw, fluid span=$m). Resampling spanwise."
        for i in 1:m
            s = 1 + (i - 1) * (nraw - 1) / max(m - 1, 1)
            i0 = floor(Int, s)
            i1 = ceil(Int, s)
            w = s - i0
            out_span[i, :] .= (1 - w) .* vals[i0, :] .+ w .* vals[i1, :]
        end
    end
    out = zeros(Float64, m, nc, 3)
    for j in 1:nc
        out[:, j, :] .= out_span
    end
    return out, nraw, false
end

decode_geometry_payload(msg, eta_span_dst::Vector{Float64}, eta_chord_dst::Vector{Float64}) =
    decode_vector_payload(msg, "geometry", eta_span_dst, eta_chord_dst)

decode_rotation_payload(msg, eta_span_dst::Vector{Float64}, eta_chord_dst::Vector{Float64}) =
    decode_vector_payload(msg, "rotation", eta_span_dst, eta_chord_dst; allow_missing=true)

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

function cross_rows(a::Array{Float64,3}, b::Array{Float64,3})
    out = zeros(Float64, size(a, 1), size(a, 2), 3)
    out[:, :, 1] .= a[:, :, 2] .* b[:, :, 3] .- a[:, :, 3] .* b[:, :, 2]
    out[:, :, 2] .= a[:, :, 3] .* b[:, :, 1] .- a[:, :, 1] .* b[:, :, 3]
    out[:, :, 3] .= a[:, :, 1] .* b[:, :, 2] .- a[:, :, 2] .* b[:, :, 1]
    return out
end



# SOCKET CONNECTION

println("Connecting to coupling server...")
sock = connect(get(ENV, "COUPLING_HOST", "127.0.0.1"), parse(Int, get(ENV, "COUPLING_PORT", "9000")))
println("Fluid connected.")
write(sock, JSON.json(Dict("role"=>"fluid")) * "\n")
flush(sock)

m_span = vlm.get_m(row_wings[1])
ys_ref = [row_wing_refs[1]._ym[i] for i in 1:m_span]
eta_span_fluid = [clamp(ys_ref[i] / span, 0.0, 1.0) for i in 1:m_span]
eta_span_coupling = build_coupling_eta_span(m_span, solid_ny_for_sampling, span_sampling_mode)
if span_sampling_mode == "custom-stride"
    assert_eta_close("Fluid VLM eta vs custom coupling eta", eta_span_fluid, eta_span_coupling)
end
eta_span_force_payload = span_sampling_mode == "custom-stride" ? eta_span_coupling : eta_span_fluid
u_prev = zeros(Float64, m_span, n_chord, 3)
omega_prev = zeros(Float64, m_span, n_chord, 3)
forces_prev = zeros(Float64, m_span, n_chord, 3)

# Geometry offsets from control-point location to other solver points.
r_vortex_ref = zeros(Float64, m_span, n_chord, 3)
r_le_ref = zeros(Float64, m_span, n_chord, 3)
r_te_ref = zeros(Float64, m_span, n_chord, 3)
for j in 1:n_chord
    wref = row_wing_refs[j]
    for i in 1:m_span
        cp = [wref._xm[i], wref._ym[i], wref._zm[i]]
        rv = [wref._xn[i] - cp[1], wref._yn[i] - cp[2], wref._zn[i] - cp[3]]
        rl = [wref._xlwingdcr[i] - cp[1], wref._ywingdcr[i] - cp[2], wref._zlwingdcr[i] - cp[3]]
        rt = [wref._xtwingdcr[i] - cp[1], wref._ywingdcr[i] - cp[2], wref._ztwingdcr[i] - cp[3]]
        r_vortex_ref[i, j, :] .= rv
        r_le_ref[i, j, :] .= rl
        r_te_ref[i, j, :] .= rt
    end
end

step_hist = Int[]
geom_res_hist = Float64[]
force_res_hist = Float64[]
force_trace_path = joinpath(save_path, run_name * "_force_payload_history.jsonl")
force_trace_io = open(force_trace_path, "w")

# Save fluid-side coupling control-point coordinates using the same flattening
# convention as payload exchange: idx=(i_span-1)*n_chord + j_chord.
fluid_cp_csv = joinpath(save_path, "fluid_coupling_cp_coords.csv")
open(fluid_cp_csv, "w") do io
    println(io, "index,i_span,j_chord,x_cp,y_cp,z_cp")
    idx = 0
    for i in 1:m_span
        for j in 1:n_chord
            idx += 1
            wref = row_wing_refs[j]
            println(
                io,
                string(
                    idx, ",", i, ",", j, ",",
                    wref._xm[i], ",", wref._ym[i], ",", wref._zm[i]
                )
            )
        end
    end
end
if DEBUG_IO
    first_ref = [row_wing_refs[1]._xm[1], row_wing_refs[1]._ym[1], row_wing_refs[1]._zm[1]]
    last_ref = [row_wing_refs[end]._xm[m_span], row_wing_refs[end]._ym[m_span], row_wing_refs[end]._zm[m_span]]
    println("FLUID REF first CP = ", first_ref)
    println("FLUID REF last  CP = ", last_ref)
    println("Saved fluid coupling CP coordinates: ", fluid_cp_csv)
end

# Receive initial geometry from solid before launching the continuous run.
msg0 = read_json_line(sock, "init")
if haskey(msg0, "dt")
    dt_solid = Float64(msg0["dt"])
    rel = abs(dt_solid - dt) / max(abs(dt), 1.0e-16)
    if rel > 1.0e-10
        @warn "Solid/fluid dt mismatch at init: solid=$(dt_solid), fluid=$(dt)"
    end
end
if haskey(msg0, "n_span")
    nspan_solid = Int(msg0["n_span"])
    nspan_solid == m_span || error(
        "Span mismatch at init: solid n_span=$(nspan_solid), fluid m_span=$(m_span)."
    )
end
if haskey(msg0, "n_chord")
    nchord_solid = Int(msg0["n_chord"])
    nchord_solid == n_chord || error(
        "Chord mismatch at init: solid n_chord=$(nchord_solid), fluid n_chord=$(n_chord)."
    )
end
if span_sampling_mode == "custom-stride" && haskey(msg0, "eta_span") && length(msg0["eta_span"]) == m_span
    eta_span_solid = Float64.(msg0["eta_span"])
    assert_eta_close("Solid eta_span vs fluid coupling eta", eta_span_solid, eta_span_coupling)
end
u_raw0, _, used2d0 = decode_geometry_payload(msg0, eta_span_fluid, eta_chord_comm)
omega_raw0, _, _ = decode_rotation_payload(msg0, eta_span_fluid, eta_chord_comm)
if DEBUG_IO
    println("RECV init first point = ", vec(u_raw0[1, 1, :]))
    println("RECV init last point  = ", vec(u_raw0[end, end, :]))
end
u_raw0[:, :, 1] .*= disp_scale_x
u_raw0[:, :, 2] .*= disp_scale_y
u_raw0[:, :, 3] .*= disp_scale_z
u_raw0 .= u_raw0 # clamp.(-max_abs_disp, max_abs_disp
omega_raw0 .= omega_raw0 #clamp.(-max_abs_rotation, max_abs_rotation
u0 = geom_relax .* u_raw0 .+ (1 - geom_relax) .* u_prev
u_prev .= u0
omega0 = geom_relax .* omega_raw0 .+ (1 - geom_relax) .* omega_prev
omega_prev .= omega0
u_vortex0 = sample_chordwise_fields(u0, eta_chord_comm, eta_chord_vortex)
u_le0 = sample_chordwise_fields(u0, eta_chord_comm, eta_chord_le)
u_te0 = sample_chordwise_fields(u0, eta_chord_comm, eta_chord_te)
omega_vortex0 = sample_chordwise_fields(omega0, eta_chord_comm, eta_chord_vortex)
omega_le0 = sample_chordwise_fields(omega0, eta_chord_comm, eta_chord_le)
omega_te0 = sample_chordwise_fields(omega0, eta_chord_comm, eta_chord_te)
u_vortex0 .+= cross_rows(omega_vortex0, r_vortex_ref)
u_le0 .+= cross_rows(omega_le0, r_le_ref)
u_te0 .+= cross_rows(omega_te0, r_te_ref)
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
use_ftot_force = Ref(true)
function ensure_gamma!(wing, m)
    if !haskey(wing.sol, "Gamma") || length(wing.sol["Gamma"]) != m
        wing.sol["Gamma"] = zeros(m)
    end
end

function safe_row_vec(sol::Dict{String, Any}, key::String, i::Int)
    if !haskey(sol, key)
        return nothing
    end
    arr = sol[key]
    if !(arr isa AbstractVector) || i > length(arr)
        return nothing
    end
    v = arr[i]
    if !(v isa AbstractVector) || length(v) < 3
        return nothing
    end
    vv = [Float64(v[1]), Float64(v[2]), Float64(v[3])]
    if any(!isfinite, vv)
        return nothing
    end
    return vv
end

function fallback_relative_velocity(wing, i::Int, T::Float64)
    Xcp = [wing._xm[i], wing._ym[i], wing._zm[i]]
    Vrel = copy(Vinf(Xcp, T))
    Vind = safe_row_vec(wing.sol, "Vind", i)
    Vvpm = safe_row_vec(wing.sol, "Vvpm", i)
    Vkin = safe_row_vec(wing.sol, "Vkin", i)
    if Vind !== nothing
        Vrel .+= Vind
    end
    if Vvpm !== nothing
        Vrel .+= Vvpm
    end
    if Vkin !== nothing
        Vrel .-= Vkin
    end
    return Vrel
end

function coupling_runtime_function(sim, PFIELD, T, DT; vprintln=(s)->nothing)
    step_ref[] += 1
    step = step_ref[]
    m = m_span
    prev_forces_snapshot = copy(forces_prev)

    if step == 1
        println("INFO: row-wing solution keys at step 1: ", collect(keys(row_wings[1].sol)))
    end

    # Extract panel forces directly from each chordwise row.
    # Use FLOWVLM force postprocessing when available (per-panel Ftot), with
    # Gamma-based fallback for robustness.
    force2d = Vector{Vector{Float64}}(undef, m * n_chord)
    for j in 1:n_chord
        ensure_gamma!(row_wings[j], m)
        gamma_j = row_wings[j].sol["Gamma"]
        for i in eachindex(gamma_j)
            circulation = gamma_j[i]
            if !isfinite(circulation)
                circulation = 0.0
            end
            gamma_j[i] = circulation #clamp -max_abs_gamma, max_abs_gamma
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
            gamma = gamma_j[i]
            if !isfinite(gamma)
                gamma = 0.0
            end
            gamma = gamma # clamp

            fx_raw, fy_raw, fz_raw = 0.0, 0.0, 0.0
            if frow != nothing && i <= length(frow)
                fi = frow[i]
                if length(fi) == 3 && all(isfinite, fi)
                    fx_raw = Float64(fi[1]) #clamp -max_abs_force, max_abs_force)
                    fy_raw = Float64(fi[2]) #clamp -max_abs_force, max_abs_force)
                    fz_raw = Float64(fi[3]) #clamp -max_abs_force, max_abs_force)
                else
                    Vloc = fallback_relative_velocity(row_wings[j], i, T)
                    lvec = [
                        row_wings[j]._xn[i+1] - row_wings[j]._xn[i],
                        row_wings[j]._yn[i+1] - row_wings[j]._yn[i],
                        row_wings[j]._zn[i+1] - row_wings[j]._zn[i],
                    ]
                    Fkj = rho * gamma * cross(Vloc, lvec)
                    fx_raw = Fkj[1] # clamp(, -max_abs_force, max_abs_force)
                    fy_raw = Fkj[2] # clamp(, -max_abs_force, max_abs_force)
                    fz_raw = Fkj[3] # clamp(, -max_abs_force, max_abs_force)
                end
            else
                Vloc = fallback_relative_velocity(row_wings[j], i, T)
                lvec = [
                    row_wings[j]._xn[i+1] - row_wings[j]._xn[i],
                    row_wings[j]._yn[i+1] - row_wings[j]._yn[i],
                    row_wings[j]._zn[i+1] - row_wings[j]._zn[i],
                ]
                Fkj = rho * gamma * cross(Vloc, lvec)
                fx_raw = Fkj[1] # clamp( , -max_abs_force, max_abs_force)
                fy_raw = Fkj[2] # clamp( , -max_abs_force, max_abs_force)
                fz_raw = Fkj[3] # clamp( , -max_abs_force, max_abs_force)
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

    # Coupling residual proxy: per-step force change.
    force_mat = reduce(vcat, (reshape(force2d[k], 1, 3) for k in 1:length(force2d)))
    prev_mat = reshape(prev_forces_snapshot, m * n_chord, 3)
    force_res = norm(force_mat - prev_mat) / max(norm(force_mat), 1.0e-16)
    push!(step_hist, step)
    push!(force_res_hist, force_res)

    println(
        force_trace_io,
        JSON.json(
            Dict(
                "step" => step,
                "n_span" => m,
                "n_chord" => n_chord,
                "indexing" => "span-major",
                "force" => force2d,
            ),
        ),
    )
    flush(force_trace_io)

    write(sock, JSON.json(Dict(
        "step"=>step,
        "n_span"=>m,
        "n_chord"=>n_chord,
        "indexing"=>"span-major",
        "dt"=>dt,
        "ttot"=>ttot,
        "eta_span"=>eta_span_force_payload,
        "eta_chord"=>eta_chord_comm,
        "force"=>force2d
    ))*"\n")
    flush(sock)

    # Receive geometry for next step and update.
    if step < nsteps
        msg = read_json_line(sock, "step $step")
        if haskey(msg, "dt")
            dt_solid = Float64(msg["dt"])
            rel = abs(dt_solid - dt) / max(abs(dt), 1.0e-16)
            if rel > 1.0e-10 && (step == 1 || step % 20 == 0)
                @warn "Solid/fluid dt mismatch at step $step: solid=$(dt_solid), fluid=$(dt)"
            end
        end
        u_raw, _, used2d = decode_geometry_payload(msg, eta_span_fluid, eta_chord_comm)
        omega_raw, _, _ = decode_rotation_payload(msg, eta_span_fluid, eta_chord_comm)
        if DEBUG_IO && (step == 1 || step % 20 == 0)
            println("RECV step=$(step) first point = ", vec(u_raw[1, 1, :]))
            println("RECV step=$(step) last point  = ", vec(u_raw[end, end, :]))
        end
        u_raw[:, :, 1] .*= disp_scale_x
        u_raw[:, :, 2] .*= disp_scale_y
        u_raw[:, :, 3] .*= disp_scale_z
        u_raw .= u_raw # clamp.(, -max_abs_disp, max_abs_disp)
        omega_raw .= omega_raw # clamp.(, -max_abs_rotation, max_abs_rotation)
        if any(!isfinite, u_raw)
            @warn "Non-finite displacement received at step $step; reusing previous geometry"
            u_raw .= u_prev
        end
        if any(!isfinite, omega_raw)
            @warn "Non-finite rotation received at step $step; reusing previous rotation"
            omega_raw .= omega_prev
        end
        u = geom_relax .* u_raw .+ (1 - geom_relax) .* u_prev
        omega = geom_relax .* omega_raw .+ (1 - geom_relax) .* omega_prev
        if step == 1 || step % 20 == 0
            println(
                "INFO: geometry magnitude at step $step: max|u|=$(maximum(abs.(u))) " *
                "max|omega|=$(maximum(abs.(omega)))"
            )
        end
        geom_res = norm(u - u_prev) / max(norm(u), 1.0e-16)
        push!(geom_res_hist, geom_res)
        u_prev .= u
        omega_prev .= omega
        u_vortex = sample_chordwise_fields(u, eta_chord_comm, eta_chord_vortex)
        u_le = sample_chordwise_fields(u, eta_chord_comm, eta_chord_le)
        u_te = sample_chordwise_fields(u, eta_chord_comm, eta_chord_te)
        omega_vortex = sample_chordwise_fields(omega, eta_chord_comm, eta_chord_vortex)
        omega_le = sample_chordwise_fields(omega, eta_chord_comm, eta_chord_le)
        omega_te = sample_chordwise_fields(omega, eta_chord_comm, eta_chord_te)
        u_vortex .+= cross_rows(omega_vortex, r_vortex_ref)
        u_le .+= cross_rows(omega_le, r_le_ref)
        u_te .+= cross_rows(omega_te, r_te_ref)
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
    shed_unsteady=use_unsteady_shedding,
    unsteady_shedcrit=unsteady_shedcrit,
    omit_shedding=omit_shedding_rows,
    wake_coupled=true,
    vlm_rlx=vlm_rlx,
    extra_runtime_function=runtime_pipeline,
    save_path=save_path,
    run_name=run_name,
    create_savepath=false,
    prompt=false,
    nsteps_save=1,
    save_horseshoes=true
)

# Save coupling diagnostics for post-processing notebook.
diag_path = joinpath(save_path, run_name * "_coupling_diagnostics.csv")
open(diag_path, "w") do io
    println(io, "step,force_residual,geometry_residual")
    n = length(step_hist)
    for k in 1:n
        gres = k <= length(geom_res_hist) ? geom_res_hist[k] : NaN
        fres = k <= length(force_res_hist) ? force_res_hist[k] : NaN
        println(io, "$(step_hist[k]),$(fres),$(gres)")
    end
end

close(sock)
close(force_trace_io)
println("Fluid solver finished.")
println("Fluid outputs saved in: $save_path")
println("Fluid diagnostics saved in: $diag_path")
println("Fluid force history saved in: $force_trace_path")
