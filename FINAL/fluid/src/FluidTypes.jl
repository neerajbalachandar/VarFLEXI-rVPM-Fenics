# Only for defining structs

struct SolverConfig

    aoa::Float64
    vinf::Float64
    rho::Float64
    nu::Float64

    ttot::Float64
    nsteps::Int
    dt::Float64

    p_per_step::Int

    lambda_vpm::Float64

    vlm_rlx::Float64

    sigma_vlm_solver::Float64
    sigma_vlm_surface::Float64

end

struct WakeConfig

    remove_every::Int

    sphere_factor::Float64

    wake_strength_min::Float64
    wake_strength_max::Float64

    wake_sigma_min::Float64
    wake_sigma_max::Float64

    shed_starting::Bool
    shed_unsteady::Bool

    unsteady_shedcrit::Float64

end

struct CouplingConfig

    host::String

    port::Int

    comm_nspan::Int

    solid_ny::Int

    span_sampling::String

    custom_stride::Union{Nothing,Int}

    geom_relax::Float64

    force_relax::Float64

end

mutable struct RuntimeState

    sock

    u_prev_cp

    omega_prev_cp

    forces_prev

    step_hist

    force_res_hist

    geom_res_hist

    force_trace_io

    step_ref

    use_ftot_force

end