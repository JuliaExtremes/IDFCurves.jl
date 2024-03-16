using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions, PDMats

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)


# Modèle marginal comme un DependentScalingModel

# test constructeur
pd = DependentScalingModel(SimpleScaling(1, 20, 5, .0, .76), UncorrelatedStructure(), IdentityCopula)
#pd = DependentScalingModel(SimpleScaling(1, 20, 5, .0, .76), UncorrelatedStructure())
#pd = DependentScalingModel(SimpleScaling(1, 20, 5, .0, .76), IdentityCopula)
#pd = DependentScalingModel(SimpleScaling(1, 20, 5, .0, .76))
# the three last commented lines : would be nice if gave same result than first line ?

# test logpdf
loglikelihood(pd, data)
loglikelihood(SimpleScaling(1, 20, 5, .0, .76), data) # same result - good

IDFCurves.fit_mle(DependentScalingModel{SimpleScaling, UncorrelatedStructure, IdentityCopula}, data, 1, [20, 5, .0, .76])
IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .0, .76]) # same result - good

mm = SimpleScaling(1, 0, 1, 0, .8)
Σ = ExponentialCorrelationStructure(1.)
pd = DependentScalingModel(mm, Σ, TCopula)
loglikelihood(pd, data)

abstract_model = DependentScalingModel{SimpleScaling, UncorrelatedStructure, IdentityCopula}
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .04, .76])


fd = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76,])
#fd = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .04, .76,1])
IDFCurves.hessian(fd, data) 

abstract_model = DependentScalingModel{SimpleScaling, UncorrelatedStructure, GaussianCopula}
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .04, .76,])
IDFCurves.hessian(fd, data) # bug
# TODO make sure that this hessian is the same as above


# Tests sur l'initialisation :

fm = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, .0, .76])
fm = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, .04, .76])
fm = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, 2*eps(), .76]) 
fm = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, 1000*eps(), .76]) 
fm = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, 1e5*eps(), .76]) 
fm = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, 1e6*eps(), .76]) 

fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76]) # renvoie résultats
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 2*eps(), .76]) # renvoie erreur
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1000*eps(), .76]) # renvoie résultat où ξ=0
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1e5*eps(), .76]) # renvoie résultat où ξ=0
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1e6*eps(), .76]) # renvoie même résultat que le premier

Distributions.score(GeneralizedExtremeValue(0,1,0))

abstract_model = DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, 2*eps(), .7, .1, 1., 1.])
params(getmarginalmodel(fd))
fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, 1000*eps(), .7, .1, 1., 1.])
params(getmarginalmodel(fd))