using IDFCurves, Test
import IDFCurves: getdistribution
using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions, PDMats, GeoStats, Random

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)



# renvoie d'un SimpleScaling model lorsque δ=0

df = CSV.read(joinpath("data","1108446.csv"), DataFrame)
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))  
data_CXH = IDFdata(df, "Year", duration_dict)

IDFCurves.fit_mle(GeneralScaling, data_CXH, 1)

# méthodes d'optimisation de fonctions :

d₀ = 1
pd = DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}
model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(pd, d₀, θ)
fobj(θ::DenseVector{<:Real}) = -IDFCurves.loglikelihood(model(θ), data)
θ₀ = IDFCurves.map_to_real_space(pd, [20, 5, .04, .76, 0.1, 1, 1])

@time IDFCurves.perform_optimization(fobj, θ₀)

# on veut comparer le temps de calcul avec :
@time optimize(fobj, θ₀, GradientDescent(), autodiff = :forward)
@time Optim.optimize(fobj, θ₀, Newton(), autodiff = :forward)
# mais ça bug





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
