"""
    TotalScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α₁::Real, α₁::Real, δ::Real, τ::Real)

Construct a TotalScaling distribution type.

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
struct TotalScaling{T<:Real, U<:Union{Real, paramfun}} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::U
    σ₀::U
    ξ::U
    α₁::U # duration exponent (defining slope of the IDF curve)
    α₂::U # duration second exponent (slope changes for different frequencies / multiscaling)
    δ::U # duration offset (defining curvature of the IDF curve)
    τ::U # intensity offset (describes analogously the flattening of the relationship for long durations)
    TotalScaling{T, U}(d₀::T, μ₀::U, σ₀::U, ξ::U, α₁::U, α₂::U, δ::U, τ::U) where {T<:Real, U<:Union{Real, paramfun}} = new{T, U}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ)
end

function TotalScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α₁::T, α₂::T, δ::T, τ::T) where {T<:Real}
        
    @assert 0 < α₁ < 1 "Scaling duration exponent must be between 0 and 1"
    # @assert α₁ < α₂ "Scaling frequency exponent must be bigger than the scaling exponent"
    @assert σ₀ > 0 "Scale must be positive"
    @assert δ ≥ 0 "Duration offset must be non-negative"
    @assert τ ≥ 0 "Intensity offset must be non-negative"

    TotalScaling{T, T}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ)
end

function TotalScaling(d₀::T, μ₀::Covariates, σ₀::Covariates, ξ::Covariates, α₁::Covariates, α₂::Covariates, δ::Covariates, τ::Covariates) where {T <: Real}
    μ₀_fun = computeparamfunction(μ₀.parameterization, standardize.(μ₀.covariates))
    σ₀_fun = computeparamfunction(σ₀.parameterization, standardize.(σ₀.covariates))
    ξ_fun = computeparamfunction(ξ.parameterization, standardize.(ξ.covariates))
    α₁_fun = computeparamfunction(α₁.parameterization, standardize.(α₁.covariates))
    α₂_fun = computeparamfunction(α₂.parameterization, standardize.(α₂.covariates))
    δ_fun = computeparamfunction(δ.parameterization, standardize.(δ.covariates))
    τ_fun = computeparamfunction(τ.parameterization, standardize.(τ.covariates))
    return TotalScaling{T, paramfun}(d₀, paramfun(μ₀.covariates, μ₀_fun, []), paramfun(σ₀.covariates, σ₀_fun, []), paramfun(ξ.covariates, ξ_fun, []), paramfun(α₁.covariates, α₁_fun, []), paramfun(α₂.covariates, α₂_fun, []), paramfun(δ.covariates, δ_fun, []), paramfun(τ.covariates, τ_fun, []))
end

function TotalScaling(d₀::T, μ₀::paramfun, σ₀::paramfun, ξ::paramfun, α₁::paramfun, α₂::paramfun, δ::paramfun, τ::paramfun) where {T <: Real}
    return TotalScaling{T, paramfun}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ)
end

TotalScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α₁::Real, α₂::Real, δ::Real, τ::Real) = TotalScaling(promote(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ)...)

Base.Broadcast.broadcastable(obj::TotalScaling) = Ref(obj)


### Parameters

"""
    duration(pd::TotalScaling)

Return the reference duration.
"""
duration(pd::TotalScaling) = pd.d₀

"""
    exponent(pd::TotalScaling)

Return the duration exponent.
"""
exponent(pd::TotalScaling) = pd.α₁
exponent_frequency(pd::TotalScaling) = pd.α₂

location(pd::TotalScaling) = pd.μ₀

"""
    offset(pd::TotalScaling)

Return the duration offset
"""
offset(pd::TotalScaling) = pd.δ

scale(pd::TotalScaling) = pd.σ₀

shape(pd::TotalScaling) = pd.ξ

intensity_offset(pd::TotalScaling) = pd.τ

params(pd::TotalScaling) = (pd.μ₀ isa paramfun ? pd.μ₀.estimators : location(pd), pd.σ₀ isa paramfun ? pd.σ₀.estimators : scale(pd), pd.ξ isa paramfun ? pd.ξ.estimators : shape(pd), pd.α₁ isa paramfun ? pd.α₁.estimators : exponent(pd), pd.α₂ isa paramfun ? pd.α₂.estimators : exponent_frequency(pd), pd.δ isa paramfun ? pd.δ.estimators : offset(pd), pd.τ isa paramfun ? pd.τ.estimators : intensity_offset(pd))

params_number(::Type{<:TotalScaling}) = 7

params_number(pd::TotalScaling) = 7 + (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0) + (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0) + (pd.δ isa paramfun ? length(pd.δ.covariate) : 0 + (pd.τ isa paramfun ? length(pd.τ.covariate) : 0))

function getcovariatenumber(pd::TotalScaling)::Int
    return sum((pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0) + (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0) + (pd.δ isa paramfun ? length(pd.δ.covariate) : 0) + (pd.τ isa paramfun ? length(pd.τ.covariate) : 0))
end

### Methods

"""
    getdistribution(pd::TotalScaling, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::TotalScaling, d::Real)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α₁ = (pd.α₁ isa paramfun ? logistic.(pd.α₁.fun(logit.(pd.α₁.estimators))) : exponent(pd)) 
    α₂ = (pd.α₂ isa paramfun ? logistic.(pd.α₂.fun(logit.(pd.α₂.estimators))) : exponent_frequency(pd))
    δ = (pd.δ isa paramfun ? exp.(pd.δ.fun(log.(pd.δ.estimators))) : offset(pd)) 
    τ = (pd.τ isa paramfun ? pd.τ.fun(pd.τ.estimators) : intensity_offset(pd))
    
    d₀ = duration(pd)
    
    ds = (log.(d .+ δ) .- log.(d₀ .+ δ))
    
    ls₁ = -α₁ .* ds
    s₁ = exp.(ls₁)

    ls₂ = -(α₁ + α₂) .* ds
    s₂ = exp.(ls₂)

    σ = (σ₀ .* s₂) # + τ

    # need to understand mu scaled...
    # best for now
    # μ = (μ₀ .* s₁) .* ((σ₀ .* s₁) + τ)
    
    # these 2 apply a sigma scaling while keeping the original u(d) formula 
    # μ = (μ₀ .* s₁ ./ σ₀) .* ((σ₀ .* s₁) + τ)
    μ = (μ₀ .* s₁) ./ σ .* ((σ₀ .* s₁) + τ)

    # confidence intervals dont work well
    # μ = μ₀ .* s₁
    # no doesnt work well, mu scaled is not simply mu0
    # μ = μ₀ ./σ  .* (σ₀ .* s₁ + τ)

    return GeneralizedExtremeValue.(μ, σ, ξ)
    
end


"""
    construct_model(::Type{<:TotalScaling}, d₀, θ)

Construct a TotalScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:TotalScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 7 "The parameter vector length must be 7. Verify that the reference duration is included."
    # println(θ)
    
    return TotalScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), logistic(θ[5]), exp(θ[6]), θ[7])

end

"""
    construct_model(::TotalScaling, d₀, θ)

Construct a TotalScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(pd::TotalScaling, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd)  "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."
    
    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α₁ = (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0)
    cov_α₂ = (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0)
    cov_δ = (pd.δ isa paramfun ? length(pd.δ.covariate) : 0)
    cov_τ = (pd.τ isa paramfun ? length(pd.τ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α₁_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α₂_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁
    δ_pos = 6 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁ + cov_α₂
    τ_pos = 7 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁ + cov_α₂ + cov_δ

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ[μ₀_pos : μ₀_pos + cov_μ₀]) : θ[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ[ξ_pos : ξ_pos + cov_ξ]) : θ[ξ_pos]
    α₁ = pd.α₁ isa paramfun ? paramfun(pd.α₁.covariate, pd.α₁.fun, logistic.(θ[α₁_pos : α₁_pos + cov_α₁])) : logistic(θ[α₁_pos])
    α₂ = pd.α₂ isa paramfun ? paramfun(pd.α₂.covariate, pd.α₂.fun, logistic.(θ[α₂_pos : α₂_pos + cov_α₂])) : logistic(θ[α₂_pos])
    δ = pd.δ isa paramfun ? paramfun(pd.δ.covariate, pd.δ.fun, exp.(θ[δ_pos : δ_pos + cov_δ])) : exp(θ[δ_pos])
    τ = pd.τ isa paramfun ? paramfun(pd.τ.covariate, pd.τ.fun, θ[τ_pos : τ_pos + cov_τ]) : θ[τ_pos]

    return getcovariatenumber(pd) > 0 ? TotalScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ) : TotalScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ)
end

"""
    construct_model(::TotalScaling, d₀, θ, c)

Construct a TotalScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(pd::TotalScaling, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == params_number(pd) "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α₁ = (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0)
    cov_α₂ = (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0)
    cov_δ = (pd.δ isa paramfun ? length(pd.δ.covariate) : 0)
    cov_τ = (pd.τ isa paramfun ? length(pd.τ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α₁_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α₂_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁
    δ_pos = 6 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁ + cov_α₂
    τ_pos = 7 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁ + cov_α₂ + cov_δ

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ_mixed[μ₀_pos : μ₀_pos + cov_μ₀]) : θ_mixed[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ_mixed[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ_mixed[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ_mixed[ξ_pos : ξ_pos + cov_ξ]) : θ_mixed[ξ_pos]
    α₁ = pd.α₁ isa paramfun ? paramfun(pd.α₁.covariate, pd.α₁.fun, logistic.(θ_mixed[α₁_pos : α₁_pos + cov_α₁])) : logistic(θ_mixed[α₁_pos])
    α₂ = pd.α₂ isa paramfun ? paramfun(pd.α₂.covariate, pd.α₂.fun, logistic.(θ_mixed[α₂_pos : α₂_pos + cov_α₂])) : logistic(θ_mixed[α₂_pos])
    δ = pd.δ isa paramfun ? paramfun(pd.δ.covariate, pd.δ.fun, exp.(θ_mixed[δ_pos : δ_pos + cov_δ])) : exp(θ_mixed[δ_pos])
    τ = pd.τ isa paramfun ? paramfun(pd.τ.covariate, pd.τ.fun, θ_mixed[τ_pos : τ_pos + cov_τ]) : θ_mixed[τ_pos]

    return getcovariatenumber(pd) > 0 ? TotalScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ) : TotalScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α₁, α₂, δ, τ)
end


"""
    map_to_real_space(::Type{<:TotalScaling}, θ)

Map the parameters from the TotalScaling parameter space to the real hypercube and return upper and lower bounds.
"""
function map_to_real_space(::Type{<:TotalScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 7 "The parameter vector length must be 7. Verify that the reference duration is included."

    @assert 0 < θ[4] < 1 "Scaling exponent must be between 0 and 1"
    # @assert θ[4] < θ[5] "Frequency scaling exponent must be bigger than the scaling exponent"
    @assert θ[2] > 0 "Scale must be positive"
    @assert θ[6] ≥ 0 "Duration offset must be non-negative"
    @assert θ[7] ≥ 0 "Intensity offset must be non-negative"

    return [θ[1], log(θ[2]), θ[3], logit(θ[4]), logit(θ[5]), log(θ[6]), θ[7]]

end

"""
    map_to_real_space(pd::TotalScaling, θ)

Map the parameters from the TotalScaling parameter space to the real hypercube.
"""
function map_to_real_space(pd::TotalScaling, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd) "The parameter vector length must be 7. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α₁ = (pd.α₁ isa paramfun ? length(pd.α₁.covariate) : 0)
    cov_α₂ = (pd.α₂ isa paramfun ? length(pd.α₂.covariate) : 0)
    cov_δ = (pd.δ isa paramfun ? length(pd.δ.covariate) : 0)
    cov_τ = (pd.τ isa paramfun ? length(pd.τ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α₁_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α₂_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁
    δ_pos = 6 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁ + cov_α₂
    τ_pos = 7 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α₁ + cov_α₂ + cov_δ
    
    @assert 0 < θ[α₁_pos] < 1 "Scaling exponent must be between 0 and 1"
    # @assert θ[α₁_pos] < θ[α₂_pos] "Frequency scaling exponent must be bigger than the scaling exponent"
    @assert θ[σ₀_pos] > 0 "Scale must be positive"
    @assert θ[δ_pos] ≥ 0 "Duration offset must be non-negative"
    @assert θ[τ_pos] ≥ 0 "Intensity offset must be non-negative"

    μ₀ = cov_μ₀ > 0 ? [θ[μ₀_pos], θ[μ₀_pos + 1 : μ₀_pos + cov_μ₀]...] : θ[μ₀_pos]
    σ₀ = cov_σ₀ > 0 ? log.([θ[σ₀_pos], θ[σ₀_pos + 1 : σ₀_pos + cov_σ₀]...]) : log(θ[σ₀_pos])
    ξ = cov_ξ > 0 ? [θ[ξ_pos], θ[ξ_pos + 1 : ξ_pos + cov_ξ]...] : θ[ξ_pos]
    α₁ = cov_α₁ > 0 ? logit.([θ[α₁_pos], θ[α₁_pos + 1 : α₁_pos + cov_α₁]...]) : logit(θ[α₁_pos])
    α₂ = cov_α₂ > 0 ? logit.([θ[α₂_pos], θ[α₂_pos + 1 : α₂_pos + cov_α₂]...]) : logit(θ[α₂_pos])
    δ = cov_δ > 0 ? log.([θ[δ_pos], θ[δ_pos + 1 : δ_pos + cov_δ]...]) : log(θ[δ_pos])
    τ = cov_τ > 0 ? [θ[τ_pos], θ[τ_pos + 1 : τ_pos + cov_τ]...] : θ[τ_pos]

    return [μ₀..., σ₀..., ξ..., α₁..., α₂..., δ..., τ...]

end

"""
    map_to_bounds(::Type{<:TotalScaling})

Return the parameter bounds.
"""
function map_to_bounds(::Type{<:TotalScaling})
    return [-Inf, 0.0001, -Inf, -Inf, -Inf, -Inf, 0.0001], [Inf, Inf, Inf, Inf, Inf, Inf, Inf]
end

"""
    map_to_bounds(::TotalScaling)

Return the parameter bounds.
"""
function map_to_bounds(pd::TotalScaling)
    return [-Inf, 0.0001, -Inf, -Inf, -Inf, -Inf, 0.0001], [Inf, Inf, Inf, Inf, Inf, Inf, Inf]
end

"""
    Base.show(io::IO, obj::TotalScaling)

Override of the show function for the objects of type TotalScaling.

"""
function Base.show(io::IO, obj::TotalScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", obj.μ₀ isa paramfun ? round.(obj.μ₀.estimators, digits=4) : round(location(obj), digits=4),
        ", σ₀ = ", obj.σ₀ isa paramfun ? round.(obj.σ₀.estimators, digits=4) : round(scale(obj), digits=4),
        ", ξ = ", obj.ξ isa paramfun ? round.(obj.ξ.estimators, digits=4) : round(shape(obj), digits=4),
        ", α₁ = ", obj.α₁ isa paramfun ? round.(obj.α₁.estimators, digits=4) : round(exponent(obj), digits=4),
        ", α₂ = ", obj.α₂ isa paramfun ? round.(obj.α₂.estimators, digits=4) : round(exponent_frequency(obj), digits=4),
        ", δ = ", obj.δ isa paramfun ? round.(obj.δ.estimators, digits=4) : round(offset(obj), digits=4),
        ", τ = ", obj.τ isa paramfun ? round.(obj.τ.estimators, digits=4) : round(intensity_offset(obj), digits=4),
        ")")
end

"""
    initialize(::Type{<:TotalScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the TotalScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::Type{<:TotalScaling}, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end

"""
    initialize(::Type{<:TotalScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the TotalScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::TotalScaling, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end