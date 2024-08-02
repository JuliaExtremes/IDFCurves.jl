using IDFCurves, Test
import IDFCurves: getdistribution
using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions, PDMats, GeoStats, Random

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)


# courant : bug lorsque le δ du General Scaling est estimé égal à 0.0


struct CompositeScaling <: MarginalScalingModel
    d₀::Real # reference duration
    μ₀::Real 
    σ₀::Real
    ξ::Real
    α_μ::Real
    α_σ::Real
end

Base.Broadcast.broadcastable(obj::CompositeScaling) = Ref(obj)

duration(pd::CompositeScaling) = pd.d₀
exponent_μ(pd::CompositeScaling) = pd.α_μ
exponent_σ(pd::CompositeScaling) = pd.α_σ
location(pd::CompositeScaling) = pd.μ₀
scale(pd::CompositeScaling) = pd.σ₀
shape(pd::CompositeScaling) = pd.ξ
params(pd::CompositeScaling) = (location(pd), scale(pd), shape(pd), exponent_μ(pd), exponent_σ(pd))
params_number(::Type{<:CompositeScaling}) = 5

function getdistribution(pd::CompositeScaling, d::Real)
    
    μ₀ = location(pd)
    σ₀ = scale(pd)
    ξ = shape(pd)
    α_μ = exponent_μ(pd)
    α_σ= exponent_σ(pd)
    
    d₀ = duration(pd)

    ls_μ = -α_μ * (log(d) - log(d₀))
    s_μ = exp(ls_μ)

    ls_σ = -α_σ * (log(d) - log(d₀))
    s_σ = exp(ls_σ)

    μ = μ₀ * s_μ
    σ = σ₀ * s_σ
    
    return GeneralizedExtremeValue(μ, σ, ξ)
    
end

durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24];
d_ref = 24; # the reference duration for parameterization will always be 24h
d_out = 1/12; # the duration for validation will always be 5min

nyear = 60

μ = 2
σ = 0.3
α = 0.7;
ξ = -0.2
gap = -0.6
α_μ = α
α_σ = α_μ + α_μ * gap
gen_model = CompositeScaling(d_ref, μ, σ, ξ, α_μ, α_σ) # model used for simulation

Random.seed!(32)
data = IDFCurves.rand(gen_model, durations, nyear);

data_wo_dout = IDFCurves.excludeduration(data, d_out)

IDFCurves.fit_mle(SimpleScaling, data_wo_dout, 1) # pas de bug
IDFCurves.fit_mle(SimpleScaling, data_wo_dout, d_out) # bug

initialize(SimpleScaling, data_wo_dout, d_out)
init_vec = initialize(SimpleScaling, data_wo_dout, 1)
param_model = SimpleScaling(d_out, init_vec...)

param_model = SimpleScaling(d_out, 90.20156951010392, 3.1925288991423924, -1.537848577652813, 0.531995477439767)
loglikelihood(param_model, data_wo_dout)

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
