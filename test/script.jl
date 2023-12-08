using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra



df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])




function logdist(x₁::Real, x₂::Real)
    @assert x₁ > 0 "point must be positive."
    @assert x₂ > 0 "point must be positive."

    return abs(log(x₁) - log(x₂))

end

@testset "logdist(x₁::Real, x₂::Real)" begin
    @test_throws AssertionError IDFCurves.logdist(-1,1)
    @test_throws AssertionError IDFCurves.logdist(1,-1)

    @test logdist(1,1) ≈ 0
    @test logdist(exp(1),exp(2)) ≈ 1
end



function logdist(x::AbstractVector{<:Real})

    T = Matrix{Float64}(undef, length(x), length(x))

    for i in eachindex(x)
        for j in eachindex(x)
            (i ≤ j) ? T[i,j] = IDFCurves.logdist(x[i], x[j]) : continue
        end
    end

    return Symmetric(T)

end

@testset "logdist(x::AbstractVector{<:Real})" begin
    x = exp.(1:2)
    @test IDFCurves.logdist(x) ≈ [0. 1.; 1. 0.]
end








# Construct the correlation matrix
IDFCurves.matern.([0. 2;2 0.],1,1)



