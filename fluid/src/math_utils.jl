function cross_rows2(a::Matrix{Float64}, b::Matrix{Float64})
    out = zeros(Float64, size(a, 1), 3)
    out[:, 1] .= a[:, 2] .* b[:, 3] .- a[:, 3] .* b[:, 2]
    out[:, 2] .= a[:, 3] .* b[:, 1] .- a[:, 1] .* b[:, 3]
    out[:, 3] .= a[:, 1] .* b[:, 2] .- a[:, 2] .* b[:, 1]
    return out
end

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
    any(!isfinite, vv) && return nothing
    return vv
end

function fallback_relative_velocity(wing, i::Int, T::Float64)
    Xcp = [wing._xm[i], wing._ym[i], wing._zm[i]]
    Vrel = copy(Vinf(Xcp, T))
    Vind = safe_row_vec(wing.sol, "Vind", i)
    Vvpm = safe_row_vec(wing.sol, "Vvpm", i)
    Vkin = safe_row_vec(wing.sol, "Vkin", i)
    Vind !== nothing && (Vrel .+= Vind)
    Vvpm !== nothing && (Vrel .+= Vvpm)
    Vkin !== nothing && (Vrel .-= Vkin)
    return Vrel
end