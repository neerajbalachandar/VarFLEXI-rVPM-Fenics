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


step_hist = Int[]
force_res_hist = Float64[]
geom_res_hist = Float64[]
force_trace_path = joinpath(save_path, run_name * "_force_payload_history.jsonl")
force_trace_io = open(force_trace_path, "w")


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