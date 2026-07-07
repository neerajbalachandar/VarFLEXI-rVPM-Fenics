runtime_pipeline = uns.concatenate(wake_treatment, coupling_runtime_function)


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