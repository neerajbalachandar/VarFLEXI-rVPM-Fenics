using Sockets
using JSON
import FLOWUnsteady as uns
import FLOWVLM as vlm


# SIMULATION PARAMETERS

AOA             = 4.2
magVinf         = 49.7
rho             = 0.93
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

Vinf(X,t) = magVinf*[cosd(AOA), 0.0, sind(AOA)]


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
    for i in 1:m
        wing._xn[i] = wing_ref._xn[i] + u_cp[i,1]
        wing._yn[i] = wing_ref._yn[i] + u_cp[i,2]
        wing._zn[i] = wing_ref._zn[i] + u_cp[i,3]
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

    # Rebuild horseshoes next call
    vlm._reset(wing; keep_Vinf=true)
end


# SOCKET CONNECTION

println("Connecting to coupling server...")
sock = connect("127.0.0.1", 9000)
println("Fluid connected.")


# TIME STEPPING LOOP

for step in 1:nsteps

    println("Fluid step $step")

    # -----------------------------------------
    # 1. RECEIVE UPDATED GEOMETRY FROM SOLID
    # -----------------------------------------
    msg = JSON.parse(String(readline(sock)))
    u_cp = reduce(hcat, msg["geometry"])'
    # u_cp must be m x 3

    # -----------------------------------------
    # 2. UPDATE WING GEOMETRY
    # -----------------------------------------
    update_geometry_absolute(wing, wing_ref, u_cp)

    # -----------------------------------------
    # 3. ADVANCE FLUID BY ONE STEP
    # -----------------------------------------
    uns.run_simulation(simulation;
        Vinf=Vinf,
        rho=rho,
        p_per_step=p_per_step,
        sigma_vlm_solver=sigma_vlm_solver,
        sigma_vlm_surf=sigma_vlm_surf,
        sigma_rotor_surf=sigma_vlm_surf,
        sigma_vpm_overwrite=sigma_vpm_overwrite,
        shed_starting=shed_starting,
        vlm_rlx=vlm_rlx
    )

    # -----------------------------------------
    # 4. EXTRACT PANEL FORCES
    # -----------------------------------------
    m = vlm.get_m(wing)
    forces = Vector{Vector{Float64}}(undef, m)

    for i in 1:m
        Γ = wing.sol["Gamma"][i]
        lift = rho * magVinf * Γ
        forces[i] = [0.0, 0.0, lift]
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