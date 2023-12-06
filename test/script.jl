using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim

import IDFCurves: location, scale, shape
import Distributions: fit_mle, params, quantile
import Extremes: fit_mle, qqplot



df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])

H = IDFCurves.hessian(fd, data)


function location(pd::dGEV, d::Real)

    d₀ = duration(pd)

    μ₀ = location(pd)

    α = exponent(pd)

    δ = offset(pd)

    ls = -α * (log(d + δ) - log(d₀ + δ))
    s = exp(ls)

    μ = μ₀ * s

    return μ

end

function scale(pd::dGEV, d::Real)

    d₀ = duration(pd)

    σ₀ = scale(pd)

    α = exponent(pd)

    δ = offset(pd)

    ls = -α * (log(d + δ) - log(d₀ + δ))
    s = exp(ls)

    σ = σ₀ * s

    return σ

end

IDFCurves.quantilevar(fd, data, 24, .95)

IDFCurves.quantilecint(fd, data, 24, .95)


function getmarginalparameters(pd::dGEV, d::Real)

    marginal = getdistribution(pd, d)

    return collect(params(marginal))

end


g(θ::DenseVector{<:Real}) = getmarginalparameters( dGEV(d₀, θ...), 24)

g(collect(params(fd)))

J = ForwardDiff.jacobian(g, collect(params(fd)))

H = IDFCurves.hessian(fd, data)


V₀ = H\J'
V = J*V₀

# function g(θ::DenseVector{<:Real}, d::Real)

#     pd = dGEV(θ...)
#     marginal = getdistribution(pd, d)

#     return [location(marginal), scale(marginal), shape(marginal)]

# end

# g([duration(fd); collect(params(fd))], 24)

# ∇(θ) = ForwardDiff.gradient(g(θ, 24), [1; collect(params(fd))])

# ∇([duration(fd); collect(params(fd))])


# fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [1, 1, 0, .9, 1])

m = getdistribution(fd, 24)
y = getdata(data, "24h")

# IDFCurves.qqplot(m, y)






quantile(fd, 1, .9)

function quantile_lbound(fd::dGEV, data::IDFdata, p::Real, α::Real=0.025)

    H = IDFCurves.hessian(fd, data)

    θ̂ = collect(params(fd))
    d₀  = duration(fd)

    g(θ::DenseVector{<:Real}) = quantile(dGEV(d₀, θ...), )

end