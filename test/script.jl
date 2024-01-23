using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)



# dGEV sans copule

fm = IDFCurves.fit_mle(dGEV, data, 1, [20, 5, .04, .76, .07])

IDFCurves.hessian(fm, data)


# dGEV avec copule et structure de covariance de Matern

model = DependentScalingModel{dGEV, MaternCorrelationStructure, GaussianCopula}

initialvalues = [20, 5, .04, .76, .07, 1., 1.]

pd = IDFCurves.fit_mle(model, data, 1, initialvalues)

# Ça ne fonctionne pas avec la différention automatique
IDFCurves.hessian(pd, data)


# Le problème provient dans la cascade du calcul de la logpdf avec la copule. Voici une version simplifiée du problème :

u = randn(9)
Σ(θ::AbstractVector{<:Real}) = cor.(MaternCorrelationStructure(θ...), h) # C'est cette opération est problématique. 
f(θ::AbstractVector{<:Real}) = logpdf(MvNormal(Σ(θ)), u)
ForwardDiff.jacobian(Σ, [1., 3.])
ForwardDiff.gradient(f, [1., 3.])

# Si on remplace la ligne problématique par sa valeur pour une copule Matern, ça fonctionne :

using BesselK, SpecialFunctions
g(ν::Real, ρ::Real) = 2^(1-ν)/SpecialFunctions.gamma(ν) * BesselK.adbesselkxv.(ν, sqrt(2*ν)*h/ρ)
f(θ::AbstractVector{<:Real}) = logpdf(MvNormal(g(θ...)), ones(9))
ForwardDiff.gradient(f, [1, 1])

# Quelques analyses supplémentaires pour le troubleshooting

# Ça ne fonctionne pas non plus pour le 1er paramètre de Matern
u = randn(9)
Σ(θ) = cor.(MaternCorrelationStructure(θ, 2.), h)
f(θ::Real) = logpdf(MvNormal(Σ(θ)), u)
ForwardDiff.derivative(Σ, 1)
ForwardDiff.derivative(f, 1)

# Ça ne fonctionne pas non plus pour le 2e paramètre de Matern
u = randn(9)
Σ(θ) = cor.(MaternCorrelationStructure(1., θ), h)
f(θ::Real) = logpdf(MvNormal(Σ(θ)), u)
ForwardDiff.derivative(Σ, 1)
ForwardDiff.derivative(f, 1)












