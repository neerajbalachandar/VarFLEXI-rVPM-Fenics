function update_geometry_absolute_spanwise!(wing, wing_ref, u_cp, u_vortex, u_le, u_te)
    m = vlm.get_m(wing)
    @assert size(u_cp) == (m, 3)
    @assert size(u_vortex) == (m, 3)
    @assert size(u_le) == (m, 3)
    @assert size(u_te) == (m, 3)

    for i in 1:m
        wing._xm[i] = wing_ref._xm[i] + u_cp[i, 1]
        wing._ym[i] = wing_ref._ym[i] + u_cp[i, 2]
        wing._zm[i] = wing_ref._zm[i] + u_cp[i, 3]
    end

    for i in 1:(m + 1)
        idx = min(i, m)
        wing._xn[i] = wing_ref._xn[i] + u_vortex[idx, 1]
        wing._yn[i] = wing_ref._yn[i] + u_vortex[idx, 2]
        wing._zn[i] = wing_ref._zn[i] + u_vortex[idx, 3]
    end

    nch = length(wing._xlwingdcr)
    for i in 1:nch
        idx = min(i, m)
        wing._xlwingdcr[i] = wing_ref._xlwingdcr[i] + u_le[idx, 1]
        wing._zlwingdcr[i] = wing_ref._zlwingdcr[i] + u_le[idx, 3]
        wing._xtwingdcr[i] = wing_ref._xtwingdcr[i] + u_te[idx, 1]
        wing._ztwingdcr[i] = wing_ref._ztwingdcr[i] + u_te[idx, 3]
        wing._ywingdcr[i] = wing_ref._ywingdcr[i] + 0.5 * (u_le[idx, 2] + u_te[idx, 2])
    end

    wing._HSs = nothing
end