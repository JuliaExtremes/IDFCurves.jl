using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, ForwardDiff, Gadfly,  LogExpFunctions, Optim

import IDFCurves.IDFdata
import Distributions: fit_mle, params, quantile
import Extremes: fit_mle, qqplot



df = CSV.read(joinpath("data","IDF_702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])
# fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [1, 1, 0, .9, 1])

m = getdistribution(fd, 24)
y = getdata(data, "24h")

# IDFCurves.qqplot(m, y)




H = hessian(fd, data)

@testset "hessian(::dGEV, data)" begin
    
    @test H ≈ [21.5015 -10.5403 46.7217 -100.863 -283.946;
    -10.5403 37.1912 21.0899 -43.1767 31.1629;
    46.7217 21.0899 1332.12 412.91 -1281.44;
   -100.863 -43.1767 412.91 21878.6 -15727.8;
   -283.946 31.1629 -1281.44 -15727.8 23588.0] rtol=.05

end

"""
    quantile(pd::dGEV, d::Real, p::Real)

Compute the quantile of level `p` for the duration `d` of the dGEV model `pd`. 
"""
function quantile(pd::dGEV, d::Real, p::Real)
    @assert 0<p<1 "The quantile level p must be in (0,1)."
    @assert d>0 "The duration must be positive."

    marginal = IDFCurves.getdistribution(pd, d)

    return quantile(marginal, p)

end

quantile(fd, 1, .9)