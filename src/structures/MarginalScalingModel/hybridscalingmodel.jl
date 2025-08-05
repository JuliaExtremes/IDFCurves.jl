struct HybridScaling{T<:Real, U<:Union{Real, paramfun}} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::U
    σ₀::U
    ξ::U
    α₁::U # small durations exponent (defining slope of the IDF curve for d <= d₀)
    α₂::U # large durations exponent (defining slope of the IDF curve for d > d₀)
    HybridScaling{T, U}(d₀::T, μ₀::U, σ₀::U, ξ::U, α₁::U, α₂::U) where {T<:Real, U<:Union{Real, paramfun}} = new{T, U}(d₀, μ₀, σ₀, ξ, α₁, α₂)
end

function HybridScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α₁::T, α₂::T) where {T<:Real}
        
    @assert 0 < α₁ < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < α₂ < 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
    
    HybridScaling{T, T}(d₀, μ₀, σ₀, ξ, α₁, α₂)
end

function HybridScaling(d₀::T, μ₀::Covariates, σ₀::Covariates, ξ::Covariates, α₁::Covariates, α₂::Covariates) where {T <: Real}
    μ₀_fun = computeparamfunction(μ₀.parameterization, standardize.(μ₀.covariates))
    σ₀_fun = computeparamfunction(σ₀.parameterization, standardize.(σ₀.covariates))
    ξ_fun = computeparamfunction(ξ.parameterization, standardize.(ξ.covariates))
    α₁_fun = computeparamfunction(α₁.parameterization, standardize.(α₁.covariates))
    α₂_fun = computeparamfunction(α₂.parameterization, standardize.(α₂.covariates))
    return HybridScaling{T, paramfun}(d₀, paramfun(μ₀.covariates, μ₀_fun, []), paramfun(σ₀.covariates, σ₀_fun, []), paramfun(ξ.covariates, ξ_fun, []), paramfun(α₁.covariates, α₁_fun, []), paramfun(α₂.covariates, α₂_fun, []))
end

function HybridScaling(d₀::T, μ₀::paramfun, σ₀::paramfun, ξ::paramfun, α₁::paramfun, α₂::paramfun) where {T <: Real}
    return HybridScaling{T, paramfun}(d₀, μ₀, σ₀, ξ, α₁, α₂)
end

HybridScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α₁::Real, α₂::Real) = HybridScaling(promote(d₀, μ₀, σ₀, ξ, α₁, α₂)...)

Base.Broadcast.broadcastable(obj::HybridScaling) = Ref(obj)


### Parameters

"""
    duration(pd::HybridScaling)

Return the reference duration.
"""
duration(pd::HybridScaling) = pd.d₀

"""
    exponent(pd::HybridScaling)

Return the location exponent.
"""
location_exponent(pd::HybridScaling) = pd.α₁

"""
    exponent(pd::HybridScaling)

Return the scale exponent.
"""
scale_exponent(pd::HybridScaling) = pd.α₂

location(pd::HybridScaling) = pd.μ₀

scale(pd::HybridScaling) = pd.σ₀

shape(pd::HybridScaling) = pd.ξ

params(pd::HybridScaling) = (pd.μ₀ isa paramfun ? pd.μ₀.estimators : location(pd), pd.σ₀ isa paramfun ? pd.σ₀.estimators : scale(pd), pd.ξ isa paramfun ? pd.ξ.estimators : shape(pd), pd.α₁ isa paramfun ? pd.α₁.estimators : location_exponent(pd), pd.α₂ isa paramfun ? pd.α₂.estimators : scale_exponent(pd))

params_number(::Type{<:HybridScaling}) = 5

params_number(pd::HybridScaling) = 5 + (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0) + (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0) 

function getcovariatenumber(pd::HybridScaling)::Int
    return sum((pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0) + (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0))
end

### Methods

"""
    getdistribution(pd::HybridScaling, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::HybridScaling, d::Real)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α₁ = (pd.α₁ isa paramfun ? logistic.(pd.α₁.fun(logit.(pd.α₁.estimators))) : location_exponent(pd)) 
    α₂ = (pd.α₂ isa paramfun ? logistic.(pd.α₂.fun(logit.(pd.α₂.estimators))) : scale_exponent(pd)) 

    d₀ = duration(pd)
    
    α = (d .≤ d₀) ? α₁ : α₂

    ls = -α .* (log.(d) .- log.(d₀))
    s = exp.(ls)

    μ = μ₀ .* s
    σ = σ₀ .* s
    
    return GeneralizedExtremeValue.(μ, σ, ξ)
    
end


"""
    construct_model(::Type{<:HybridScaling}, d₀, θ)

Construct a HybridScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:HybridScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is included."
    
    return HybridScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), logistic(θ[5]))

end

"""
    construct_model(::Type{<:HybridScaling}, d₀, θ, c)

Construct a HybridScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:HybridScaling}, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == 5 "The parameter vector length must be 5. Verify that the reference duration is included."

    return HybridScaling(d₀, θ_mixed[1], exp(θ_mixed[2]), θ_mixed[3], logistic(θ_mixed[4]), logistic(θ_mixed[5]))

end

"""
    construct_model(::HybridScaling, d₀, θ)

Construct a HybridScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(pd::HybridScaling, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd)  "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."
    
    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α₁ = (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0)
    cov_α₂ = (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α₁_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α₂_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ[μ₀_pos : μ₀_pos + cov_μ₀]) : θ[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ[ξ_pos : ξ_pos + cov_ξ]) : θ[ξ_pos]
    α₁ = pd.α₁ isa paramfun ? paramfun(pd.α₁.covariate, pd.α₁.fun, logistic.(θ[α₁_pos : α₁_pos + cov_α₁])) : logistic(θ[α₁_pos])
    α₂ = pd.α₂ isa paramfun ? paramfun(pd.α₂.covariate, pd.α₂.fun, logistic.(θ[α₂_pos : α₂_pos + cov_α₂])) : logistic(θ[α₂_pos])

    return getcovariatenumber(pd) > 0 ? HybridScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α₁, α₂) : HybridScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α₁, α₂)
end

"""
    construct_model(::HybridScaling, d₀, θ, c)

Construct a HybridScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(pd::HybridScaling, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == params_number(pd) "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α₁ = (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0)
    cov_α₂ = (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α₁_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α₂_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ_mixed[μ₀_pos : μ₀_pos + cov_μ₀]) : θ_mixed[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ_mixed[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ_mixed[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ_mixed[ξ_pos : ξ_pos + cov_ξ]) : θ_mixed[ξ_pos]
    α₁ = pd.α₁ isa paramfun ? paramfun(pd.α₁.covariate, pd.α₁.fun, logistic.(θ_mixed[α₁_pos : α₁_pos + cov_α₁])) : logistic(θ_mixed[α₁_pos])
    α₂ = pd.α₂ isa paramfun ? paramfun(pd.α₂.covariate, pd.α₂.fun, logistic.(θ_mixed[α₂_pos : α₂_pos + cov_α₂])) : logistic(θ_mixed[α₂_pos])

    return getcovariatenumber(pd) > 0 ? HybridScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α₁, α₂) : HybridScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α₁, α₂)
end


"""
    map_to_real_space(::Type{<:HybridScaling}, θ)

Map the parameters from the HybridScaling parameter space to the real hypercube.
"""
function map_to_real_space(::Type{<:HybridScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is included."

    @assert 0 < θ[4] < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < θ[5] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[2] > 0 "Scale must be positive"

    return [θ[1], log(θ[2]), θ[3], logit(θ[4]), logit(θ[5])]

end

"""
    map_to_real_space(pd::HybridScaling, θ)

Map the parameters from the HybridScaling parameter space to the real hypercube.
"""
function map_to_real_space(pd::HybridScaling, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd) "The parameter vector length must be 5. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α₁ = (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0)
    cov_α₂ = (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α₁_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α₂_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁
    
    @assert 0 < θ[α₁_pos] < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < θ[α₂_pos] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[σ₀_pos] > 0 "Scale must be positive"

    μ₀ = cov_μ₀ > 0 ? [θ[μ₀_pos], θ[μ₀_pos + 1 : μ₀_pos + cov_μ₀]...] : θ[μ₀_pos]
    σ₀ = cov_σ₀ > 0 ? log.([θ[σ₀_pos], θ[σ₀_pos + 1 : σ₀_pos + cov_σ₀]...]) : log(θ[σ₀_pos])
    ξ = cov_ξ > 0 ? [θ[ξ_pos], θ[ξ_pos + 1 : ξ_pos + cov_ξ]...] : θ[ξ_pos]
    α₁ = cov_α₁ > 0 ? logit.([θ[α₁_pos], θ[α₁_pos + 1 : α₁_pos + cov_α₁]...]) : logit(θ[α₁_pos])
    α₂ = cov_α₂ > 0 ? logit.([θ[α₂_pos], θ[α₂_pos + 1 : α₂_pos + cov_α₂]...]) : logit(θ[α₂_pos])

    return [μ₀..., σ₀..., ξ..., α₁..., α₂...]

end

"""
    Base.show(io::IO, obj::HybridScaling)

Override of the show function for the objects of type HybridScaling.

"""
function Base.show(io::IO, obj::HybridScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", obj.μ₀ isa paramfun ? round.(obj.μ₀.estimators, digits=4) : round(location(obj), digits=4),
        ", σ₀ = ", obj.σ₀ isa paramfun ? round.(obj.σ₀.estimators, digits=4) : round(scale(obj), digits=4),
        ", ξ = ", obj.ξ isa paramfun ? round.(obj.ξ.estimators, digits=4) : round(shape(obj), digits=4),
        ", α₁ = ", obj.α₁ isa paramfun ? round.(obj.α₁.estimators, digits=4) : round(location_exponent(obj), digits=4),
        ", α₂ = ", obj.α₂ isa paramfun ? round.(obj.α₂.estimators, digits=4) : round(scale_exponent(obj), digits=4),
        ")")
end

"""
    initialize(::Type{<:HybridScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the HybridScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::Type{<:HybridScaling}, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end

"""
    initialize(::Type{<:HybridScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the HybridScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::HybridScaling, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end