using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])


θ₀ = IDFCurves.map_to_real_space(dGEV, [20, 5, .04, .76, .07])

fobj(θ::DenseVector{<:Real}) =  -loglikelihood(dGEV(1, IDFCurves.map_to_param_space(dGEV, θ)...), data)

res = Optim.optimize(TwiceDifferentiable(fobj, θ₀; autodiff = :forward), θ₀)









durations = getduration.(data, gettag(data))
h = IDFCurves.logdist(durations)

model(θ::DenseVector{<:Real}) = DependentScalingModel(
    dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
    MvTDist(15, IDFCurves.matern.(h, exp(θ[6]), exp(θ[7])))
    )

model([1,0,0,.8,.5,-2,-1])

fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)


θ̂ = params(fd)

initialvalues = [θ̂[1], log(θ̂[2]), logit(θ̂[3])-.5, log(θ̂[4]), log(θ̂[5]), log(1.), log(1.)]

# res = Optim.optimize(TwiceDifferentiable(fobj, initialvalues; autodiff = :forward), initialvalues)
res = Optim.optimize(fobj, initialvalues)







fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [1, 1, 0, .9, 1])


h = IDFCurves.logdist(durations)
Σ = IDFCurves.matern.(h, 5, 1) 
C = MvTDist(15, Σ)

fmm = DependentScalingModel(fd, C)















# Construct the correlation matrix
IDFCurves.matern.([0. 2;2 0.],1,1)



