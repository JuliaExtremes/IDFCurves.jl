using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions, PDMats, GeoStats

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)
getyear(data, "5min")



# courant : (procédure de test)


pd_type = SimpleScaling
d_out = minimum(values(getduration(data)))
q=100


# First step : parameter estimation
train_data = IDFCurves.excludeduration(data, d_out)
fitted_model = IDFCurves.fit_mle(pd_type, train_data, d_out)

# Fisher information matrix (normalized)
hess = IDFCurves.hessian(fitted_model, train_data)
norm_I_Fisher = hess / length(IDFCurves.getyear(data, IDFCurves.gettag(data,d_out)))

# Test statistic
distrib_theo_d_out = IDFCurves.getdistribution(fitted_model, d_out) # attention fitted_model doit être un marginalscalingmodel -> à modifier lorsuqe arg sera un DependentScalingModel.
stat = IDFCurves.cvmcriterion(distrib_theo_d_out, IDFCurves.getdata(data, IDFCurves.gettag(data,d_out)))

# Kernel function ρ
g = IDFCurves.get_g(fitted_model, d_out)
g(0.8)
ρ(u,v) = minimum([u,v]) - u*v + g(u)' * ( norm_I_Fisher \ g(v) )

function zolotarev_approx1(λs::Vector{<:Real} , x::Real)
    
    q = length(λs)

    λ₁ = λs[1]
    
    term1 = - sum([ 0.5 * log(1 - λs[i]/λ₁) for i in 2:q])
    term2 = 1/SpecialFunctions.gamma(0.5)
    term3 = ( x/(2*λ₁) )^(-0.5)
    term4 = exp( - (x/(2*λ₁)) )
        
    return maximum([1 - exp(term1) * term2 * term3 * term4, 0 ])
end

λs = IDFCurves.approx_eigenvalues(ρ, 100)
zolotarev_approx1(λs , stat)

λs = fill(1, 100)
zolotarev_approx1(λs , 11)
quantile(Chi(100), 0.95)

function zolotarev_approx2(λs::Vector{<:Real} , x::Real)
    @assert length(λs) >= 1 "The vector of λ must contain at elast one element for the Zolotarev approximation to be valid."

    # taking multiplicity into account
    γs = unique(λs)
    multiplicities = [count(==(γ), λs) for γ in γs]

    q = length(γs)
    γ₁ = γs[1]
    
    term1 = - sum([ 0.5 * multiplicities[i] * log(1 - γs[i]/γ₁) for i in 2:q])
    term2 = - log(SpecialFunctions.gamma(0.5 * multiplicities[1]))
    term3 = (0.5 * multiplicities[1] - 1) * log( x/(2*γ₁) )
    term4 = - (x/(2*γ₁)) 

    if term1 < term4
        @warn "Zolotarev approximation is outside its validity domain. No conclusion can be made from small p-value."
    end
        
    return maximum([1 - exp(term1 + term2 + term3 + term4), 0 ])
end

λs = fill(1, 10)
zolotarev_approx2(λs , 1)
zolotarev_approx2(λs , quantile(Chisq(10), 0.99))
zolotarev_approx2(λs , quantile(Chisq(10), 0.01))

λs = Float64[]
for k in 1:10
    append!(λs, fill(1/2^k, 2*k^2))
end

#λs = IDFCurves.approx_eigenvalues(ρ, 100)
#λs = fill(1, 100)
sample = Float64[]
for i in 1:10000
    val = 0
    for λ in λs
        val += λ * rand(Normal(0,1))^2
    end
    push!(sample,val)
end
quantile(sample, 0.999)

zolotarev_approx2(λs , quantile(sample, 0.99))
zolotarev_approx2([1] , quantile(Chisq(1), 0.99))
zolotarev_approx1(λs , quantile(sample, 0.99))

λs = fill(1, 100)
zolotarev_approx2(λs , 11)
quantile(Chi(100), 0.95)


fd = IDFCurves.SimpleScaling(1, 10, 2, 0.1, 0.7)
g = IDFCurves.get_g(fd, 1)

g(0)
g(0.8)
f(θ::AbstractArray) = [exp(θ[1]), exp(θ[2]), IDFCurves.logistic(θ[3])-.5, IDFCurves.logistic(θ[4])]
jac = ForwardDiff.jacobian(f, [log(IDFCurves.location(fd)), log(IDFCurves.scale(fd)), IDFCurves.logit(IDFCurves.shape(fd)+0.5), IDFCurves.logit(IDFCurves.exponent(fd))])
jac*g(0.8)
# méthodes d'optimisation de fonctions :

g = IDFCurves.get_g(fd, 2)
g(0.8)
g(0.5)

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
