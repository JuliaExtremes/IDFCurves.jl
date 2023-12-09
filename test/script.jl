using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions

import Distributions: quantile, cdf



df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])

"""
    cdf(pd::dGEV, d::Real, x::Real)

Return the cdf of the marginal distribution for duration `d` of the model `pd` evaluated at `x`.
"""
function cdf(pd::dGEV, d::Real, x::Real)

    margdist = IDFCurves.getdistribution(pd, d)
    return cdf(margdist, x)

end

function cdf(pd::dGEV, d::Real, x::AbstractVector{<:Real})

    margdist = IDFCurves.getdistribution(pd, d)
    return cdf.(margdist, x)

end


@testset "cdf(::dGEV)" begin
    pd = dGEV(1, 100, 1, 0, .8, 5)

    @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1 , 0), 100)
    @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1 , 0), [100, 200])
end





v = fill(10, 1000)

@time cdf.(fd, 1, v)

@time cdfv(fd, 1, v)



h = IDFCurves.logdist(durations)
Σ = IDFCurves.matern.(h, 5, 1) 
C = MvTDist(15, Σ)



m = length(C)
u = rand(m)


@time logpdf_TCopula(C, u)

uv = [rand(m) for i in 1:10]

@time logpdf_TCopula.(Ref(C), uv)







# Construct the correlation matrix
IDFCurves.matern.([0. 2;2 0.],1,1)



