struct CompositeScaling{T<:Real, U<:Union{Real, paramfun}} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::U
    σ₀::U
    ξ::U
    α_μ::U # location exponent (defining slope of the IDF curve for μ)
    α_σ::U # shape exponent (defining slope of the IDF curve for σ)
    CompositeScaling{T, U}(d₀::T, μ₀::U, σ₀::U, ξ::U, α_μ::U, α_σ::U) where {T<:Real, U<:Union{Real, paramfun}} = new{T, U}(d₀, μ₀, σ₀, ξ, α_μ, α_σ)
end

function CompositeScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α_μ::T, α_σ::T) where {T<:Real}
        
    @assert 0 < α_μ < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < α_σ < 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
    
    CompositeScaling{T, T}(d₀, μ₀, σ₀, ξ, α_μ, α_σ)
end

function CompositeScaling(d₀::T, μ₀::Covariates, σ₀::Covariates, ξ::Covariates, α_μ::Covariates, α_σ::Covariates) where {T <: Real}
    μ₀_fun = computeparamfunction(μ₀.parameterization, standardize.(μ₀.covariates))
    σ₀_fun = computeparamfunction(σ₀.parameterization, standardize.(σ₀.covariates))
    ξ_fun = computeparamfunction(ξ.parameterization, standardize.(ξ.covariates))
    α_μ_fun = computeparamfunction(α_μ.parameterization, standardize.(α_μ.covariates))
    α_σ_fun = computeparamfunction(α_σ.parameterization, standardize.(α_σ.covariates))
    return CompositeScaling{T, paramfun}(d₀, paramfun(μ₀.covariates, μ₀_fun, []), paramfun(σ₀.covariates, σ₀_fun, []), paramfun(ξ.covariates, ξ_fun, []), paramfun(α_μ.covariates, α_μ_fun, []), paramfun(α_σ.covariates, α_σ_fun, []))
end

function CompositeScaling(d₀::T, μ₀::paramfun, σ₀::paramfun, ξ::paramfun, α_μ::paramfun, α_σ::paramfun) where {T <: Real}
    return CompositeScaling{T, paramfun}(d₀, μ₀, σ₀, ξ, α_μ, α_σ)
end

CompositeScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α_μ::Real, α_σ::Real) = CompositeScaling(promote(d₀, μ₀, σ₀, ξ, α_μ, α_σ)...)

Base.Broadcast.broadcastable(obj::CompositeScaling) = Ref(obj)


### Parameters

"""
    duration(pd::CompositeScaling)

Return the reference duration.
"""
duration(pd::CompositeScaling) = pd.d₀

"""
    exponent(pd::CompositeScaling)

Return the location exponent.
"""
location_exponent(pd::CompositeScaling) = pd.α_μ

"""
    exponent(pd::CompositeScaling)

Return the scale exponent.
"""
scale_exponent(pd::CompositeScaling) = pd.α_σ

location(pd::CompositeScaling) = pd.μ₀

scale(pd::CompositeScaling) = pd.σ₀

shape(pd::CompositeScaling) = pd.ξ

params(pd::CompositeScaling) = (pd.μ₀ isa paramfun ? pd.μ₀.estimators : location(pd), pd.σ₀ isa paramfun ? pd.σ₀.estimators : scale(pd), pd.ξ isa paramfun ? pd.ξ.estimators : shape(pd), pd.α_μ isa paramfun ? pd.α_μ.estimators : location_exponent(pd), pd.α_σ isa paramfun ? pd.α_σ.estimators : scale_exponent(pd))

params_number(::Type{<:CompositeScaling}) = 5

params_number(pd::CompositeScaling) = 5 + (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α_μ isa paramfun ? length(pd.α_μ.covariate) : 0) + (pd.α_σ isa paramfun ? length(pd.α_σ.covariate) : 0) 

function getcovariatenumber(pd::CompositeScaling)::Int
    return sum((pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α_μ isa paramfun ? length(pd.α_μ.covariate) : 0) + (pd.α_σ isa paramfun ? length(pd.α_σ.covariate) : 0))
end

### Methods

"""
    getdistribution(pd::CompositeScaling, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::CompositeScaling, d::Real, isPrint=false)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α_μ = (pd.α_μ isa paramfun ? logistic.(pd.α_μ.fun(logit.(pd.α_μ.estimators))) : location_exponent(pd)) 
    α_σ = (pd.α_σ isa paramfun ? logistic.(pd.α_σ.fun(logit.(pd.α_σ.estimators))) : scale_exponent(pd)) 
    
    d₀ = duration(pd)
    
    # ls = log.(d .+ δ) .- log.(d₀ .+ δ)
    ls = log.(d) .- log.(d₀)
    α1 = -α_μ .* ls
    s_μ = exp.(α1)
    μ = μ₀ .* s_μ

    α2 = -α_σ .* ls
    s_σ = exp.(α2)
    σ = σ₀ .* s_σ
    
    return GeneralizedExtremeValue.(μ, σ, ξ)
    
end


"""
    construct_model(::Type{<:CompositeScaling}, d₀, θ)

Construct a CompositeScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:CompositeScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 6. Verify that the reference duration is included."
    
    return CompositeScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), logistic(θ[5]))

end

"""
    construct_model(::CompositeScaling, d₀, θ)

Construct a CompositeScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(pd::CompositeScaling, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd)  "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."
    
    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α_μ = (pd.α_μ isa paramfun ? length(pd.α_μ.covariate) : 0)
    cov_α_σ = (pd.α_σ isa paramfun ? length(pd.α_σ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_μ_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α_σ_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α_μ

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ[μ₀_pos : μ₀_pos + cov_μ₀]) : θ[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ[ξ_pos : ξ_pos + cov_ξ]) : θ[ξ_pos]
    α_μ = pd.α_μ isa paramfun ? paramfun(pd.α_μ.covariate, pd.α_μ.fun, logistic.(θ[α_μ_pos : α_μ_pos + cov_α_μ])) : logistic(θ[α_μ_pos])
    α_σ = pd.α_σ isa paramfun ? paramfun(pd.α_σ.covariate, pd.α_σ.fun, logistic.(θ[α_σ_pos : α_σ_pos + cov_α_σ])) : logistic(θ[α_σ_pos])

    return getcovariatenumber(pd) > 0 ? CompositeScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α_μ, α_σ) : CompositeScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α_μ, α_σ)
end

"""
    construct_model(::CompositeScaling, d₀, θ, c)

Construct a CompositeScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(pd::CompositeScaling, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == params_number(pd) "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α_μ = (pd.α_μ isa paramfun ? length(pd.α_μ.covariate) : 0)
    cov_α_σ = (pd.α_σ isa paramfun ? length(pd.α_σ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_μ_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α_σ_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α_μ

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ_mixed[μ₀_pos : μ₀_pos + cov_μ₀]) : θ_mixed[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ_mixed[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ_mixed[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ_mixed[ξ_pos : ξ_pos + cov_ξ]) : θ_mixed[ξ_pos]
    α_μ = pd.α_μ isa paramfun ? paramfun(pd.α_μ.covariate, pd.α_μ.fun, logistic.(θ_mixed[α_μ_pos : α_μ_pos + cov_α_μ])) : logistic(θ_mixed[α_μ_pos])
    α_σ = pd.α_σ isa paramfun ? paramfun(pd.α_σ.covariate, pd.α_σ.fun, logistic.(θ_mixed[α_σ_pos : α_σ_pos + cov_α_σ])) : logistic(θ_mixed[α_σ_pos])

    return getcovariatenumber(pd) > 0 ? CompositeScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α_μ, α_σ) : CompositeScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α_μ, α_σ)
end


"""
    map_to_real_space(::Type{<:CompositeScaling}, θ)

Map the parameters from the CompositeScaling parameter space to the real hypercube.
"""
function map_to_real_space(::Type{<:CompositeScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 6. Verify that the reference duration is included."

    @assert 0 < θ[4] < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < θ[5] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[2] > 0 "Scale must be positive"

    return [θ[1], log(θ[2]), θ[3], logit(θ[4]), logit(θ[5])]

end

"""
    map_to_real_space(pd::CompositeScaling, θ)

Map the parameters from the CompositeScaling parameter space to the real hypercube.
"""
function map_to_real_space(pd::CompositeScaling, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd) "The parameter vector length must be 5. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α_μ = (pd.α_μ isa paramfun ? length(pd.α_μ.covariate) : 0)
    cov_α_σ = (pd.α_σ isa paramfun ? length(pd.α_σ.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_μ_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    α_σ_pos = 5 + cov_μ₀ + cov_σ₀ + cov_ξ + cov_α_μ
    
    @assert 0 < θ[α_μ_pos] < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < θ[α_σ_pos] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[σ₀_pos] > 0 "Scale must be positive"

    μ₀ = cov_μ₀ > 0 ? [θ[μ₀_pos], θ[μ₀_pos + 1 : μ₀_pos + cov_μ₀]...] : θ[μ₀_pos]
    σ₀ = cov_σ₀ > 0 ? log.([θ[σ₀_pos], θ[σ₀_pos + 1 : σ₀_pos + cov_σ₀]...]) : log(θ[σ₀_pos])
    ξ = cov_ξ > 0 ? [θ[ξ_pos], θ[ξ_pos + 1 : ξ_pos + cov_ξ]...] : θ[ξ_pos]
    α_μ = cov_α_μ > 0 ? logit.([θ[α_μ_pos], θ[α_μ_pos + 1 : α_μ_pos + cov_α_μ]...]) : logit(θ[α_μ_pos])
    α_σ = cov_α_σ > 0 ? logit.([θ[α_σ_pos], θ[α_σ_pos + 1 : α_σ_pos + cov_α_σ]...]) : logit(θ[α_σ_pos])

    return [μ₀..., σ₀..., ξ..., α_μ..., α_σ...]

end

"""
    Base.show(io::IO, obj::CompositeScaling)

Override of the show function for the objects of type CompositeScaling.

"""
function Base.show(io::IO, obj::CompositeScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", obj.μ₀ isa paramfun ? round.(obj.μ₀.estimators, digits=4) : round(location(obj), digits=4),
        ", σ₀ = ", obj.σ₀ isa paramfun ? round.(obj.σ₀.estimators, digits=4) : round(scale(obj), digits=4),
        ", ξ = ", obj.ξ isa paramfun ? round.(obj.ξ.estimators, digits=4) : round(shape(obj), digits=4),
        ", α_μ = ", obj.α_μ isa paramfun ? round.(obj.α_μ.estimators, digits=4) : round(location_exponent(obj), digits=4),
        ", α_σ = ", obj.α_σ isa paramfun ? round.(obj.α_σ.estimators, digits=4) : round(scale_exponent(obj), digits=4),
        ")")
end

"""
    initialize(::Type{<:CompositeScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the CompositeScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::Type{<:CompositeScaling}, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end

"""
    initialize(::Type{<:CompositeScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the CompositeScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the SImpleScaling model. δ is initialized at (close to) 0 as a default.
"""
function initialize(::CompositeScaling, data::IDFdata, d₀::Real)
    
    init_simple_scaling = initialize(SimpleScaling, data, d₀)

    return [ init_simple_scaling ; [0.001] ]

end