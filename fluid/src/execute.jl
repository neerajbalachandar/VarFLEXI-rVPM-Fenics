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
    vpm_viscous = vpm_viscous,
    shed_starting=shed_starting,
    shed_unsteady=use_unsteady_shedding,
    unsteady_shedcrit=unsteady_shedcrit,
    omit_shedding=omit_shedding_rows,
    wake_coupled=wake_coupled,
    vlm_rlx=vlm_rlx,
    extra_runtime_function=runtime_pipeline,
    save_path=save_path,
    run_name=run_name,
    create_savepath=create_savepath,
    prompt=prompt,
    nsteps_save=nsteps_save,
    save_horseshoes=save_horseshoes
)

open(diag_path, "w") do io
    println(io, "step,force_residual,geometry_residual,lift,drag,cl,cd,fluid_step_time")
    n = length(step_hist)
    for k in 1:n
        gres = k <= length(geom_res_hist) ? geom_res_hist[k] : NaN
        fres = k <= length(force_res_hist) ? force_res_hist[k] : NaN
        lift = k <= length(lift_hist) ? lift_hist[k] : NaN
        drag = k <= length(drag_hist) ? drag_hist[k] : NaN
        cl = k <= length(cl_hist) ? cl_hist[k] : NaN
        cd = k <= length(cd_hist) ? cd_hist[k] : NaN
        step_time = k <= length(fluid_step_time_hist) ? fluid_step_time_hist[k] : NaN
        println(io, "$(step_hist[k]),$(fres),$(gres),$(lift),$(drag),$(cl),$(cd),$(step_time)")
    end
end

function disconnect_server(sock)

    close(sock)

    println("Fluid socket closed.")

end


disconnect_server(sock)
close(force_trace_io)
println("Fluid v9 finished.")
println("Outputs: $save_path")
println("Diagnostics: $diag_path")
println("Force trace: $force_trace_path")
