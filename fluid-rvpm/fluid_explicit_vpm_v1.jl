# Classic working code without geomtry inclusions, flow field vis, or chordwise discretization

using Sockets
using JSON
using LinearAlgebra
import FLOWUnsteady as uns
import FLOWVLM as vlm

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


# SIMULATION PARAMETERS
AOA             = 0
magVinf         = 10
rho             = 0.10
b               = 2.489
ar              = 5.0
tr              = 1.0
twist_root      = 0.0
twist_tip       = 0.0
lambda          = 0.0
gamma           = 0.0

n               = 50
r               = 10.0
central         = false

wakelength      = 2.75*b
ttot            = wakelength/magVinf
nsteps          = 200
dt              = ttot/nsteps

# VPM parameters
p_per_step      = 1
lambda_vpm      = 2.0
sigma_vpm_overwrite = lambda_vpm * magVinf * dt / p_per_step
sigma_vlm_solver = -1
sigma_vlm_surf   = 0.05*b
shed_starting    = true
vlm_rlx          = 0.7

# Coupling stabilization (numerical damping)
geom_relax       = 0.20           # 0<geom_relax<=1; lower is more damping
force_relax      = 0.20           # 0<force_relax<=1
max_abs_disp     = 0.01*b         # clamp incoming displacement magnitude
max_abs_force    = 1.0e6          # clamp outgoing per-panel force component
disp_scale_x     = 0.0            # keep 0 to avoid chord distortion instability
disp_scale_y     = 0.0            # keep 0 to avoid spanwise panel collapse
disp_scale_z     = 1.0

Vinf(X,t) = magVinf*[1.0, 0.0, 0.0] # 1cos(10t)


# Primary Question - why do we have a geometry definition here? if the solid fenics has a geom use the same, why confuse?
# GEOMETRY
println("Initializing geometry...")

wing = vlm.simpleWing(b, ar, tr, twist_root, lambda, gamma;
                        twist_tip=twist_tip, n=n, r=r, central=central)

wing_ref = deepcopy(wing)

system = vlm.WingSystem()
vlm.addwing(system, "Wing", wing)

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
run_name = "fluid_explicit_vpm_v1"
mkpath(save_path)

# Maximum number of particles
max_particles = (nsteps+1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step+1) + p_per_step)

# GEOMETRY UPDATE
function update_geometry_absolute(wing, wing_ref, u_cp)

    m = vlm.get_m(wing)

    @assert size(u_cp,1) == m
    @assert size(u_cp,2) == 3

    # --- Control points ---
    for i in 1:m
        wing._xm[i] = wing_ref._xm[i] + u_cp[i,1]
        wing._ym[i] = wing_ref._ym[i] + u_cp[i,2]
        wing._zm[i] = wing_ref._zm[i] + u_cp[i,3]
    end

    # --- Bound vortices ---
    # NOTE: bound-vortex points have length m+1, while u_cp has m control points.
    # Clamp the last point to the last control-point displacement.
    for i in 1:(m+1)
        idx = min(i, m)
        wing._xn[i] = wing_ref._xn[i] + u_cp[idx,1]
        wing._yn[i] = wing_ref._yn[i] + u_cp[idx,2]
        wing._zn[i] = wing_ref._zn[i] + u_cp[idx,3]
    end

    # --- Leading & trailing edges ---
    nch = length(wing._xlwingdcr)

    for i in 1:nch
        idx = min(i, m)

        wing._xlwingdcr[i] = wing_ref._xlwingdcr[i] + u_cp[idx,1]
        wing._ywingdcr[i]  = wing_ref._ywingdcr[i]  + u_cp[idx,2]
        wing._zlwingdcr[i] = wing_ref._zlwingdcr[i] + u_cp[idx,3]

        wing._xtwingdcr[i] = wing_ref._xtwingdcr[i] + u_cp[idx,1]
        wing._ztwingdcr[i] = wing_ref._ztwingdcr[i] + u_cp[idx,3]
    end

    # Rebuild horseshoes next call without clearing wing.sol["Gamma"].
    # FLOWUnsteady expects previous-step Gamma during precalculations.
    wing._HSs = nothing
end



# SOCKET CONNECTION

println("Connecting to coupling server...")
sock = connect("127.0.0.1", 9000)
println("Fluid connected.")
write(sock, JSON.json(Dict("role"=>"fluid")) * "\n")
flush(sock)

m_global = vlm.get_m(wing)
u_cp_prev = zeros(Float64, m_global, 3)
forces_prev = [zeros(3) for _ in 1:m_global]


# TIME STEPPING LOOP

for step in 1:nsteps

    println("Fluid step $step")

    # -----------------------------------------
    # 1. RECEIVE UPDATED GEOMETRY FROM SOLID
    # -----------------------------------------
    msg = JSON.parse(String(readline(sock)))
    geo = msg["geometry"]
    m = vlm.get_m(wing)

    ng = length(geo)
    if ng == 0
        error("Received empty geometry array from coupling")
    end

    u_cp_raw = zeros(Float64, m, 3)
    if ng == m
        for i in 1:m
            u_cp_raw[i,1] = Float64(geo[i][1])
            u_cp_raw[i,2] = Float64(geo[i][2])
            u_cp_raw[i,3] = Float64(geo[i][3])
        end
    else
        @warn "Geometry panel count mismatch (solid=$ng, fluid=$m). Resampling."
        if ng == 1
            for i in 1:m
                u_cp_raw[i,1] = Float64(geo[1][1])
                u_cp_raw[i,2] = Float64(geo[1][2])
                u_cp_raw[i,3] = Float64(geo[1][3])
            end
        else
            for i in 1:m
                s = 1 + (i-1) * (ng-1) / max(m-1, 1)
                i0 = floor(Int, s)
                i1 = ceil(Int, s)
                w = s - i0
                for k in 1:3
                    v0 = Float64(geo[i0][k])
                    v1 = Float64(geo[i1][k])
                    u_cp_raw[i,k] = (1-w)*v0 + w*v1
                end
            end
        end
    end
    u_cp_raw[:,1] .*= disp_scale_x
    u_cp_raw[:,2] .*= disp_scale_y
    u_cp_raw[:,3] .*= disp_scale_z
    if any(!isfinite, u_cp_raw)
        @warn "Non-finite displacement received; reusing previous geometry"
        u_cp_raw .= u_cp_prev
    end
    u_cp_raw .= clamp.(u_cp_raw, -max_abs_disp, max_abs_disp)
    u_cp = geom_relax .* u_cp_raw .+ (1 - geom_relax) .* u_cp_prev
    u_cp_prev .= u_cp

    # -----------------------------------------
    # 2. UPDATE WING GEOMETRY
    # -----------------------------------------
    update_geometry_absolute(wing, wing_ref, u_cp)
    if !haskey(wing.sol, "Gamma")
        wing.sol["Gamma"] = zeros(m)
    end

    # -----------------------------------------
    # 3. ADVANCE FLUID BY ONE STEP
    # -----------------------------------------
    uns.run_simulation(simulation, 1;
        Vinf=Vinf,
        rho=rho,
        p_per_step=p_per_step,
        max_particles=max_particles,
        sigma_vlm_solver=sigma_vlm_solver,
        sigma_vlm_surf=sigma_vlm_surf,
        sigma_rotor_surf=sigma_vlm_surf,
        sigma_vpm_overwrite=sigma_vpm_overwrite,
        shed_starting=(step == 1 ? shed_starting : false),
        vlm_rlx=vlm_rlx,
        save_path=save_path,
        run_name=run_name,
        create_savepath=false,
        prompt=false,
        nsteps_save=1
    )

    # -----------------------------------------
    # 4. EXTRACT PANEL FORCES
    # -----------------------------------------
    m = vlm.get_m(wing)
    forces = Vector{Vector{Float64}}(undef, m)

    for i in 1:m
        Γ = wing.sol["Gamma"][i]
        if !isfinite(Γ)
            Γ = 0.0
        end
        lift = rho * magVinf * Γ
        fz = clamp(lift, -max_abs_force, max_abs_force)
        forces[i] = [0.0, 0.0, fz]
    end

    # Force relaxation before sending to solid
    for i in 1:m
        for k in 1:3
            forces[i][k] = force_relax*forces[i][k] + (1-force_relax)*forces_prev[i][k]
        end
        forces_prev[i] = copy(forces[i])
    end

    println("DEBUG: First force = ", forces[1])

    # -----------------------------------------
    # 5. SEND FORCES TO COUPLING SERVER
    # -----------------------------------------
    write(sock, JSON.json(Dict("step"=>step, "force"=>forces))*"\n")
    flush(sock)

end

close(sock)
println("Fluid solver finished.")
println("Fluid outputs saved in: $save_path")
