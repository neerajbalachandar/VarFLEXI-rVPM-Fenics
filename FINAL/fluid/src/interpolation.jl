function interp_profile(eta_src::Vector{Float64}, vals::Matrix{Float64}, eta::Float64)
    n = length(eta_src)
    n == 1 && return copy(vals[1, :])

    e = clamp(eta, eta_src[1], eta_src[end])
    j = searchsortedlast(eta_src, e)
    if j <= 0
        return copy(vals[1, :])
    elseif j >= n
        return copy(vals[end, :])
    end

    e0 = eta_src[j]
    e1 = eta_src[j + 1]
    w = (e - e0) / max(e1 - e0, eps(Float64))
    return (1 - w) .* vals[j, :] .+ w .* vals[j + 1, :]
end

function sample_span_chord(grid::Array{Float64, 3}, eta_span_src::Vector{Float64}, eta_chord_src::Vector{Float64},
                           eta_span_dst::Vector{Float64}, eta_chord_q::Float64)
    if size(grid, 1) == 0
        return zeros(Float64, length(eta_span_dst), 3)
    end

    ns = size(grid, 1)
    tmp = zeros(Float64, ns, 3)
    for i in 1:ns
        vals = reshape(grid[i, :, :], size(grid, 2), 3)
        tmp[i, :] .= interp_profile(eta_chord_src, vals, eta_chord_q)
    end

    out = zeros(Float64, length(eta_span_dst), 3)
    for i in eachindex(eta_span_dst)
        out[i, :] .= interp_profile(eta_span_src, tmp, eta_span_dst[i])
    end
    return out
end

uniform_eta(n::Int) = n <= 1 ? [0.0] : collect(range(0.0, 1.0; length=n))