using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions, PDMats, GeoStats

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)
getyear(data, "5min")


# courant (initialisation automatique)


initialize(ExponentialCorrelationStructure, data)


initialize(MaternCorrelationStructure, data)

pd = DependentScalingModel{SimpleScaling, ExponentialCorrelationStructure, GaussianCopula}
fd = IDFCurves.fit_mle(pd, data, 1, [20, 5, .04, .76, 1]) # tout se passe bien

collect(params(fd))
collect(params(getcorrelogram(fd)))

ρ(h) = cor( getcorrelogram(fd), h)

layer1 = layer(ρ, 0, 6, Theme(default_color = "red"))
layer2 = layer(kendall_data, x=:distance, y=:kendall)
plot(layer1, layer2)

pd = DependentScalingModel{SimpleScaling, MaternCorrelationStructure, GaussianCopula}
fd = IDFCurves.fit_mle(pd, data, 1, [20, 5, .04, .76, 1, 1]) # tout se passe bien

ρ(h) = cor( getcorrelogram(fd), h)

layer1 = layer(ρ, 0, 6, Theme(default_color = "red"))
layer2 = layer(kendall_data, x=:distance, y=:kendall)
plot(layer1, layer2)


# Tests sur l'estimation

# On ne peut pas utiliser la différentitation automatique avec une copule de Student

abstract_model = DependentScalingModel{SimpleScaling, UncorrelatedStructure, TCopula{1}}
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .04, .76]) # Passe à l'optimisation sans gradient
IDFCurves.hessian(fd, data) # bug # Crée bug 


# Tests sur l'initialisation :

fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76]) # renvoie résultats
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 2*eps(), .76]) # renvoie meme resultat que le premier
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1000*eps(), .76]) # renvoie résultat où ξ=0
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1e5*eps(), .76]) # renvoie erreur
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1e6*eps(), .76]) # renvoie même résultat que le premier
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 0., .76]) # renvoie même résultat que le premier

# Jonathan :

Distributions.score(GeneralizedExtremeValue(0,1,0))

abstract_model = DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, 2*eps(), .7, .1, 1., 1.])
params(getmarginalmodel(fd))
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, 1000*eps(), .7, .1, 1., 1.])
params(getmarginalmodel(fd))
