# Are we gonna use this?

module Logger

using CSV
using DataFrames

mutable struct FluidHistory

    step::Vector{Int}
    time::Vector{Float64}

    Fx::Vector{Float64}
    Fy::Vector{Float64}
    Fz::Vector{Float64}

    force_residual::Vector{Float64}
    geometry_residual::Vector{Float64}

    particles::Vector{Int}

    runtime::Vector{Float64}

end

function FluidHistory()

    FluidHistory(

        Int[],

        Float64[],

        Float64[],

        Float64[],

        Float64[],

        Float64[],

        Float64[],

        Int[],

        Float64[]

    )

end

function save(history,path)

    CSV.write(path,DataFrame(history))

end

end