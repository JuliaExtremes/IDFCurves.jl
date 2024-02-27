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

# Ça fonctionne avec la différention automatique !
IDFCurves.hessian(pd, data)

# # Les lignes de code suivantes peuvent être exécutées et produisent le résultat escompté :
# u = randn(9)
# Σ(θ::AbstractVector{<:Real}) = cor.(MaternCorrelationStructure(θ...), h) 
# f(θ::AbstractVector{<:Real}) = logpdf(MvNormal(Σ(θ)), u)
# ForwardDiff.hessian(f, [1.5, 3.2]) 


struct ExpCorrStruct{T<:Real} <: AbstractCorrelationStructure
    θ::T
    function ExpCorrStruct(θ::T) where {T<:Real} 
        @assert θ > 0 "exponential correlogram parameter must be positive"        
        return new{T}(θ)
    end
end

ExpCorrStruct(d)  # ça fonctionne


# # Par contre on peut reproduire le bug en faisant la chose suivante :
# initial_model = IDFCurves.construct_model(DependentScalingModel{dGEV, MaternCorrelationStructure, GaussianCopula}, data, 24, [20, 5, .04, .76, .07, 1., 1.])
# create_model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(typeof(initial_model), data, 24, θ)
# fobj(θ::DenseVector{<:Real}) = -loglikelihood(create_model(θ), data)
# H = ForwardDiff.hessian(fobj, [20, 5, .04, .76, .07, 1., 1.])


# # Le bug a l'air causé par le fait que le type renvoyé par "typeof(initial_model)" fait intervenir MaternCorrelationStructure{Float64},
# # et donc on ne peut plus remplacer θ par des ForwardDiff.Dual
# typeof(initial_model)

# # ici on n'observe pas de bug :
# create_model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(DependentScalingModel{dGEV, MaternCorrelationStructure, GaussianCopula}, data, 24, θ)
# fobj(θ::DenseVector{<:Real}) = -loglikelihood(create_model(θ), data)
# H = ForwardDiff.hessian(fobj, [20, 5, .04, .76, .07, 1., 1.])

# # on veut donc que la fonction getcorrelogramtype() renvoie MaternCorrelationStructure et pas MaternCorrelationStructure{Float64} 
# # D'où la transformation effectuée qui a éliminé le bug.

# typeof(initial_model)