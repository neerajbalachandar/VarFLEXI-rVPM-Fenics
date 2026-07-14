function build_coupling_eta_span(nspan::Int, ny_solid::Int, mode::String; custom_stride::Union{Nothing,Int}=nothing)
    nspan < 1 && error("COUPLING_NSPAN_COMM/FLUID_N_SPAN must be >= 1")
    nspan == 1 && return [0.0]

    if mode == "midpoint"
        return [(i - 0.5) / nspan for i in 1:nspan]
    elseif mode == "node-stride"
        stride = max(1, ny_solid ÷ nspan)
        idx = [(i - 1) * stride for i in 1:nspan]
        if length(unique(idx)) < nspan || maximum(idx) > ny_solid
            idx = round.(Int, range(0, max(ny_solid - 1, 0); length=nspan))
        end
        return [idx_i / max(float(ny_solid), 1.0) for idx_i in idx]
    elseif mode == "custom-stride"
        custom_stride === nothing &&
            error("COUPLING_SPAN_SAMPLING=custom-stride requires COUPLING_SPAN_STRIDE")
        stride = custom_stride
        stride < 1 && error("COUPLING_SPAN_STRIDE must be >= 1")
        last_idx = (nspan - 1) * stride
        last_idx > ny_solid && error(
            "Unsafe custom span stride: " *
            "(COUPLING_NSPAN_COMM-1)*COUPLING_SPAN_STRIDE = $(last_idx) " *
            "exceeds SOLID_NY = $(ny_solid). Reduce stride/stations or increase SOLID_NY."
        )
        return [((i - 1) * stride) / max(float(ny_solid), 1.0) for i in 1:nspan]
    elseif mode == "linspace"
        return [(i - 1) / nspan for i in 1:nspan]
    end

    error("Unsupported COUPLING_SPAN_SAMPLING=$(mode)")
end

function assert_eta_close(name::String, a::Vector{Float64}, b::Vector{Float64}; mode::String="", tol=1.0e-10)
    length(a) == length(b) || error("$(name) length mismatch: $(length(a)) vs $(length(b))")
    err = maximum(abs.(a .- b))
    err <= tol || error(
        "$(name) mismatch for COUPLING_SPAN_SAMPLING=$(mode): " *
        "max |eta_a-eta_b| = $(err). The fluid VLM station locations must match " *
        "the coupling eta grid; otherwise forces are labeled at the wrong span stations."
    )
end
