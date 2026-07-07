function apply_from_message!(msg; first_step::Bool=false, step::Int=0)
    u_le, u_te, u_cp, u_bv, used_edge_payload = extract_le_te_cp_bv(msg, eta_span_fluid)
    omega_le, omega_te, omega_cp, omega_bv = extract_rotations(msg, eta_span_fluid)

    # User-controlled scaling for debug/consistency sweeps.
    u_cp[:, 1] .*= disp_scale_x
    u_cp[:, 2] .*= disp_scale_y
    u_cp[:, 3] .*= disp_scale_z
    u_le[:, 1] .*= disp_scale_x
    u_le[:, 2] .*= disp_scale_y
    u_le[:, 3] .*= disp_scale_z
    u_te[:, 1] .*= disp_scale_x
    u_te[:, 2] .*= disp_scale_y
    u_te[:, 3] .*= disp_scale_z
    u_bv[:, 1] .*= disp_scale_x
    u_bv[:, 2] .*= disp_scale_y
    u_bv[:, 3] .*= disp_scale_z

    # Relax only CP displacement state; LE/TE/BV follow this step's message.
    u_cp .= geom_relax .* u_cp .+ (1 - geom_relax) .* u_prev_cp
    omega_cp .= geom_relax .* omega_cp .+ (1 - geom_relax) .* omega_prev_cp

    # Reconstruct LE/TE/BV from relaxed CP when edge payload is unavailable.
    if !used_edge_payload
        u_le .= u_cp
        u_te .= u_cp
        u_bv .= u_cp
    end

    # Rotational correction at each geometry point.
    # CP has no offset from itself; keep direct displacement at CP.
    u_bv .+= cross_rows2(omega_bv, r_vortex_ref)
    u_le .+= cross_rows2(omega_le, r_le_ref)
    u_te .+= cross_rows2(omega_te, r_te_ref)

    update_geometry_absolute_spanwise!(wing, wing_ref, u_cp, u_bv, u_le, u_te)
    ensure_gamma!(wing, m_span)

    u_prev_cp .= u_cp
    omega_prev_cp .= omega_cp

    if DEBUG_IO && (first_step || step == 1 || step % 20 == 0)
        println("RECV geometry mode = ", used_edge_payload ? "LE/TE direct" : "legacy geometry->sampled")
        println("RECV first CP/LE/TE = ", vec(u_cp[1, :]), " / ", vec(u_le[1, :]), " / ", vec(u_te[1, :]))
        println("RECV last  CP/LE/TE = ", vec(u_cp[end, :]), " / ", vec(u_le[end, :]), " / ", vec(u_te[end, :]))
    end
end