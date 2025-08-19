"""
    GeneralScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real)

Construct a GeneralScaling distribution type.

## Details

**TODO**

``\\mu_d = \\mu_0 \\left(\\frac{d+\\delta}{d_0+\\delta} \\right) ^{-\\alpha} \\qquad 
  \\sigma_d = \\sigma_0 \\left(\\frac{d+\\delta}{d_0+\\delta} \\right) ^{-\\alpha} \\qquad 
  \\xi_d = \\xi``

## References

Koutsoyiannis, D., Kozonis, D. and Manetas, A. (1998). 
A mathematical framework for studying rainfall intensity-duration-frequency relationships,
*Journal of Hydrology*, 206(1-2), 118-135, https://doi.org/10.1016/S0022-1694(98)00097-3.
"""
struct GeneralScaling{T<:Real, U<:Union{Real, paramfun}} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::U
    σ₀::U
    ξ::U
    α::U # duration exponent (defining slope of the IDF curve)
    δ::U # duration offset (defining curvature of the IDF curve)
    GeneralScaling{T, U}(d₀::T, μ₀::U, σ₀::U, ξ::U, α::U, δ::U) where {T<:Real, U<:Union{Real, paramfun}} = new{T, U}(d₀, μ₀, σ₀, ξ, α, δ)
end

function GeneralScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T) where {T<:Real}
        
    @assert 0 < α < 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
    @assert δ ≥ 0 "Duration offset must be non-negative"
    
    GeneralScaling{T, T}(d₀, μ₀, σ₀, ξ, α, δ)
end

function GeneralScaling(d₀::T, μ₀::Covariates, σ₀::Covariates, ξ::Covariates, α::Covariates, δ::Covariates) where {T <: Real}
    μ₀_fun = computeparamfunction(μ₀.parameterization, standardize.(μ₀.covariates))
    σ₀_fun = computeparamfunction(σ₀.parameterization, standardize.(σ₀.covariates))
    ξ_fun = computeparamfunction(ξ.parameterization, standardize.(ξ.covariates))
    α_fun = computeparamfunction(α.parameterization, standardize.(α.covariates))
    δ_fun = computeparamfunction(δ.parameterization, standardize.(δ.covariates))
    return GeneralScaling{T, paramfun}(d₀, paramfun(μ₀.covariates, μ₀_fun, []), paramfun(σ₀.covariates, σ₀_fun, []), paramfun(ξ.covariates, ξ_fun, []), paramfun(α.covariates, α_fun, []), paramfun(δ.covariates, δ_fun, []))
end

function GeneralScaling(d₀::T, μ₀::paramfun, σ₀::paramfun, ξ::paramfun, α::paramfun, δ::paramfun) where {T <: Real}
    return GeneralScaling{T, paramfun}(d₀, μ₀, σ₀, ξ, α, δ)
end

GeneralScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real) = GeneralScaling(promote(d₀, μ₀, σ₀, ξ, α, δ)...)

Base.Broadcast.broadcastable(obj::GeneralScaling) = Ref(obj)


### Parameters

"""
    duration(pd::GeneralScaling)

Return the reference duration.
"""
duration(pd::GeneralScaling) = pd.d₀

"""
    exponent(pd::GeneralScaling)

Return the duration exponent.
"""
exponent(pd::GeneralScaling) = pd.α

location(pd::GeneralScaling) = pd.μ₀

"""
    offset(pd::GeneralScaling)

Return the duration offset
"""
offset(pd::GeneralScaling) = pd.δ

scale(pd::GeneralScaling) = pd.σ₀

shape(pd::GeneralScaling) = pd.ξ

params(pd::GeneralScaling) = (pd.μ₀ isa paramfun ? pd.μ₀.estimators : location(pd), pd.σ₀ isa paramfun ? pd.σ₀.estimators : scale(pd), pd.ξ isa paramfun ? pd.ξ.estimators : shape(pd), pd.α isa paramfun ? pd.α.estimators : exponent(pd), pd.δ isa paramfun ? pd.δ.estimators : offset(pd))

params_number(::Type{<:GeneralScaling}) = 5

params_number(pd::GeneralScaling) = 5 + (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α isa paramfun ? length(pd.α.covariate) : 0) + (pd.δ isa paramfun ? length(pd.δ.covariate) : 0) 

function getcovariatenumber(pd::GeneralScaling)::Int
    return sum((pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α isa paramfun ? length(pd.α.covariate) : 0) + (pd.δ isa paramfun ? length(pd.δ.covariate) : 0))
end

### Methods

"""
    getdistribution(pd::GeneralScaling, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::GeneralScaling, d::Real)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α = (pd.α isa paramfun ? logistic.(pd.α.fun(logit.(pd.α.estimators))) : exponent(pd)) 
    δ = (pd.δ isa paramfun ? exp.(pd.δ.fun(log.(pd.δ.estimators))) : offset(pd)) 
    
    d₀ = duration(pd)
    
    ls = -α .* (log.(d .+ δ) .- log.(d₀ .+ δ))
    s = exp.(ls)

    μ = μ₀ .* s
    σ = σ₀ .* s
    
    return GeneralizedExtremeValue.(μ, σ, ξ)
    
end


"""
    getquantile(pd::GeneralScaling, t::Real, d::Real)

Return the quantile for duration `d` and return level `t`.
"""
function getquantile(pd::GeneralScaling, t::Real, d::Real)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α = (pd.α isa paramfun ? logistic.(pd.α.fun(logit.(pd.α.estimators))) : exponent(pd)) 
    δ = (pd.δ isa paramfun ? exp.(pd.δ.fun(log.(pd.δ.estimators))) : offset(pd)) 

    p = (1 .- 1 ./ t)
    ls = (-log(p)).^(-ξ)
    
    d₀ = duration(pd)
    
    ls = -α .* (log.(d .+ δ) .- log.(d₀ .+ δ))
    s = exp.(ls)
    
    return (μ₀ .+ ((σ₀ ./ ξ) .* ((-log(p)).^(-ξ) .- 1))) .* s
end


"""
    construct_model(::Type{<:GeneralScaling}, d₀, θ)

Construct a GeneralScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:GeneralScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is included."
    
    return GeneralScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), exp(θ[5]))

end

"""
    construct_model(::Type{<:GeneralScaling}, d₀, θ, c)

Construct a GeneralScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(::Type{<:GeneralScaling}, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == 5 "The parameter vector length must be 5. Verify that the reference duration is not included."
    return GeneralScaling(d₀, θ_mixed[1], exp(θ_mixed[2]), θ_mixed[3], logistic(θ_mixed[4]), exp(θ_mixed[5]))
end

"""
    construct_model(::GeneralScaling, d₀, θ)

Construct a GeneralScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(pd::GeneralScaling, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd)  "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."
    
    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α = (pd.α isa paramfun ? length(pd.α.covariate) : 0)
    cov_δ = (pd.δ isa paramfun ? length(pd.δ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    δ_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ[μ₀_pos : μ₀_pos + cov_μ₀]) : θ[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ[ξ_pos : ξ_pos + cov_ξ]) : θ[ξ_pos]
    α = pd.α isa paramfun ? paramfun(pd.α.covariate, pd.α.fun, logistic.(θ[α_pos : α_pos + cov_α])) : logistic(θ[α_pos])
    δ = pd.δ isa paramfun ? paramfun(pd.δ.covariate, pd.δ.fun, exp.(θ[δ_pos : δ_pos + cov_δ])) : exp(θ[δ_pos])

    return getcovariatenumber(pd) > 0 ? GeneralScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α, δ) : GeneralScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α, δ)
end

"""
    construct_model(::GeneralScaling, d₀, θ, c)

Construct a GeneralScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(pd::GeneralScaling, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == params_number(pd) "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α = (pd.α isa paramfun ? length(pd.α.covariate) : 0)
    cov_δ = (pd.δ isa paramfun ? length(pd.δ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    δ_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ_mixed[μ₀_pos : μ₀_pos + cov_μ₀]) : θ_mixed[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ_mixed[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ_mixed[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ_mixed[ξ_pos : ξ_pos + cov_ξ]) : θ_mixed[ξ_pos]
    α = pd.α isa paramfun ? paramfun(pd.α.covariate, pd.α.fun, logistic.(θ_mixed[α_pos : α_pos + cov_α])) : logistic(θ_mixed[α_pos])
    δ = pd.δ isa paramfun ? paramfun(pd.δ.covariate, pd.δ.fun, exp.(θ_mixed[δ_pos : δ_pos + cov_δ])) : exp(θ_mixed[δ_pos])

    return getcovariatenumber(pd) > 0 ? GeneralScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α, δ) : GeneralScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α, δ)
end


"""
    map_to_real_space(::Type{<:GeneralScaling}, θ)

Map the parameters from the GeneralScaling parameter space to the real hypercube.
"""
function map_to_real_space(::Type{<:GeneralScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is included."

    @assert 0 < θ[4] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[2] > 0 "Scale must be positive"
    @assert θ[5] ≥ 0 "Duration offset must be non-negative"

    return [θ[1], log(θ[2]), θ[3], logit(θ[4]), log(θ[5])]
end

"""
    map_to_real_space(pd::GeneralScaling, θ)

Map the parameters from the GeneralScaling parameter space to the real hypercube.
"""
function map_to_real_space(pd::GeneralScaling, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd) "The parameter vector length must be 5. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α = (pd.α isa paramfun ? length(pd.α.covariate) : 0)
    cov_δ = (pd.δ isa paramfun ? length(pd.δ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    δ_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α
    
    @assert 0 < θ[α_pos] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[σ₀_pos] > 0 "Scale must be positive"
    @assert θ[δ_pos] ≥ 0 "Duration offset must be non-negative"

    μ₀ = cov_μ₀ > 0 ? [θ[μ₀_pos], θ[μ₀_pos + 1 : μ₀_pos + cov_μ₀]...] : θ[μ₀_pos]
    σ₀ = cov_σ₀ > 0 ? log.([θ[σ₀_pos], θ[σ₀_pos + 1 : σ₀_pos + cov_σ₀]...]) : log(θ[σ₀_pos])
    ξ = cov_ξ > 0 ? [θ[ξ_pos], θ[ξ_pos + 1 : ξ_pos + cov_ξ]...] : θ[ξ_pos]
    α = cov_α > 0 ? logit.([θ[α_pos], θ[α_pos + 1 : α_pos + cov_α]...]) : logit(θ[α_pos])
    δ = cov_δ > 0 ? log.([θ[δ_pos], θ[δ_pos + 1 : δ_pos + cov_δ]...]) : log(θ[δ_pos])

    return [μ₀..., σ₀..., ξ..., α..., δ...]

end

"""
    Base.show(io::IO, obj::GeneralScaling)

Override of the show function for the objects of type GeneralScaling.

"""
function Base.show(io::IO, obj::GeneralScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", obj.μ₀ isa paramfun ? round.(obj.μ₀.estimators, digits=4) : round(location(obj), digits=4),
        ", σ₀ = ", obj.σ₀ isa paramfun ? round.(obj.σ₀.estimators, digits=4) : round(scale(obj), digits=4),
        ", ξ = ", obj.ξ isa paramfun ? round.(obj.ξ.estimators, digits=4) : round(shape(obj), digits=4),
        ", α = ", obj.α isa paramfun ? round.(obj.α.estimators, digits=4) : round(exponent(obj), digits=4),
        ", δ = ", obj.δ isa paramfun ? round.(obj.δ.estimators, digits=4) : round(offset(obj), digits=4),
        ")")
end

"""
    initialize(::Type{<:GeneralScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the GeneralScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::Type{<:GeneralScaling}, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end

"""
    initialize(::Type{<:GeneralScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the GeneralScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::GeneralScaling, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end