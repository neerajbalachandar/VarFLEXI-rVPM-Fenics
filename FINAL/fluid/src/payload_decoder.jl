function decode_vector_payload(msg, key::String; allow_missing::Bool=false)
    if !haskey(msg, key)
        allow_missing && return zeros(Float64, 0, 0, 3), Float64[], Float64[]
        error("Payload missing key \"$key\"")
    end

    raw = msg[key]
    nraw = length(raw)
    if nraw == 0
        allow_missing && return zeros(Float64, 0, 0, 3), Float64[], Float64[]
        error("Received empty payload array for key \"$key\"")
    end

    vals = zeros(Float64, nraw, 3)
    for i in 1:nraw
        vals[i, 1] = Float64(raw[i][1])
        vals[i, 2] = Float64(raw[i][2])
        vals[i, 3] = Float64(raw[i][3])
    end

    n_span_in = haskey(msg, "n_span") ? Int(msg["n_span"]) : nraw
    n_chord_in = haskey(msg, "n_chord") ? Int(msg["n_chord"]) : 1
    if n_span_in < 1 || n_chord_in < 1 || n_span_in * n_chord_in != nraw
        eta_span = uniform_eta(nraw)
        eta_chord = [0.0]
        grid = reshape(copy(vals), nraw, 1, 3)
        return grid, eta_span, eta_chord
    end

    eta_span = haskey(msg, "eta_span") && length(msg["eta_span"]) == n_span_in ?
               Float64.(msg["eta_span"]) : uniform_eta(n_span_in)
    eta_chord = haskey(msg, "eta_chord") && length(msg["eta_chord"]) == n_chord_in ?
                Float64.(msg["eta_chord"]) : uniform_eta(n_chord_in)

    p_s = sortperm(eta_span)
    p_c = sortperm(eta_chord)
    eta_span = eta_span[p_s]
    eta_chord = eta_chord[p_c]

    indexing = haskey(msg, "indexing") ? String(msg["indexing"]) : "span-major"
    grid = zeros(Float64, n_span_in, n_chord_in, 3)

    idx = 1
    if indexing == "span-major"
        for i in 1:n_span_in
            for j in 1:n_chord_in
                grid[i, j, :] .= vals[idx, :]
                idx += 1
            end
        end
    elseif indexing == "chord-major"
        for j in 1:n_chord_in
            for i in 1:n_span_in
                grid[i, j, :] .= vals[idx, :]
                idx += 1
            end
        end
    else
        @warn "Unknown indexing='$indexing' for key '$key'; defaulting to span-major"
        for i in 1:n_span_in
            for j in 1:n_chord_in
                grid[i, j, :] .= vals[idx, :]
                idx += 1
            end
        end
    end

    sorted_grid = similar(grid)
    for i in 1:n_span_in, j in 1:n_chord_in
        sorted_grid[i, j, :] .= grid[p_s[i], p_c[j], :]
    end

    return sorted_grid, eta_span, eta_chord
end

function extract_le_te_cp_bv(msg, eta_span_dst)
    # Preferred protocol: dedicated LE/TE payloads from solid.
    if haskey(msg, "geometry_le") && haskey(msg, "geometry_te")
        g_le, eta_s_le, eta_c_le = decode_vector_payload(msg, "geometry_le")
        g_te, eta_s_te, eta_c_te = decode_vector_payload(msg, "geometry_te")
        u_le = sample_span_chord(g_le, eta_s_le, eta_c_le, eta_span_dst, 0.0)
        u_te = sample_span_chord(g_te, eta_s_te, eta_c_te, eta_span_dst, 1.0)

        # Reconstruct CP/BV points from communicated edges.
        u_cp = (1 - eta_cp) .* u_le .+ eta_cp .* u_te
        u_bv = (1 - eta_bv) .* u_le .+ eta_bv .* u_te

        return u_le, u_te, u_cp, u_bv, true
    end

    # Backward-compatible protocol: use generic "geometry" payload and sample at needed chord etas.
    g, eta_s, eta_c = decode_vector_payload(msg, "geometry")
    u_le = sample_span_chord(g, eta_s, eta_c, eta_span_dst, 0.0)
    u_te = sample_span_chord(g, eta_s, eta_c, eta_span_dst, 1.0)
    u_cp = sample_span_chord(g, eta_s, eta_c, eta_span_dst, eta_cp)
    u_bv = sample_span_chord(g, eta_s, eta_c, eta_span_dst, eta_bv)
    return u_le, u_te, u_cp, u_bv, false
end

function extract_rotations(msg, eta_span_dst)
    if haskey(msg, "rotation_le") && haskey(msg, "rotation_te")
        r_le, eta_s_le, eta_c_le = decode_vector_payload(msg, "rotation_le")
        r_te, eta_s_te, eta_c_te = decode_vector_payload(msg, "rotation_te")
        omega_le = sample_span_chord(r_le, eta_s_le, eta_c_le, eta_span_dst, 0.0)
        omega_te = sample_span_chord(r_te, eta_s_te, eta_c_te, eta_span_dst, 1.0)
        omega_cp = (1 - eta_cp) .* omega_le .+ eta_cp .* omega_te
        omega_bv = (1 - eta_bv) .* omega_le .+ eta_bv .* omega_te
        return omega_le, omega_te, omega_cp, omega_bv
    end

    rot, eta_s, eta_c = decode_vector_payload(msg, "rotation"; allow_missing=true)
    if size(rot, 1) == 0
        z = zeros(Float64, length(eta_span_dst), 3)
        return z, z, z, z
    end
    omega_le = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, 0.0)
    omega_te = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, 1.0)
    omega_cp = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, eta_cp)
    omega_bv = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, eta_bv)
    return omega_le, omega_te, omega_cp, omega_bv
end