using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)


# Simple scaling sans copule

fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76])

H = IDFCurves.hessian(fm, data)
Hmoins1 = inv(H)
params_conf_intervals = [(params(fm)[i] - quantile(Normal(0,1), 0.975) * sqrt(Hmoins1[i,i]), params(fm)[i] + quantile(Normal(0,1), 0.975) * sqrt(Hmoins1[i,i])) for i in 1:4]


# dGEV sans copule

fm = IDFCurves.fit_mle(dGEV, data, 1, [20, 5, .04, .76, .7])

H = IDFCurves.hessian(fm, data)
Hmoins1 = inv(H)
params_conf_intervals = [(params(fm)[i] - quantile(Normal(0,1), 0.975) * sqrt(Hmoins1[i,i]), params(fm)[i] + quantile(Normal(0,1), 0.975) * sqrt(Hmoins1[i,i])) for i in 1:5]


# dGEV avec copule et structure de covariance de Matern

model = DependentScalingModel{dGEV, MaternCorrelationStructure, GaussianCopula}

initialvalues = [20, 5, .04, .76, .07, 1., 1.]

pd = IDFCurves.fit_mle(model, data, 1, initialvalues)

# Tests sur l'initialisation :

fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76]) # renvoie résultats
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 2*eps(), .76]) # renvoie erreur
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1000*eps(), .76]) # renvoie résultat où ξ=0
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 0.0001, .76]) # renvoie même résultat que le premier