"""
    SimpleScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real)

Construct a simple scaling distribution type.

## Details

**TODO**

``\\mu_d = \\mu_0 \\left(\\frac{d}{d_0} \\right) ^{-\\alpha} \\qquad 
  \\sigma_d = \\sigma_0 \\left(\\frac{d}{d_0} \\right) ^{-\\alpha} \\qquad 
  \\xi_d = \\xi``

## References

Koutsoyiannis, D., Kozonis, D. and Manetas, A. (1998). 
A mathematical framework for studying rainfall intensity-duration-frequency relationships,
*Journal of Hydrology*, 206(1-2), 118-135, https://doi.org/10.1016/S0022-1694(98)00097-3.
"""
struct SimpleScaling{T<:Real, U<:Union{Real, paramfun}} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::U 
    σ₀::U
    ξ::U
    α::U # scaling exponent (defining slope of the IDF curve)
    SimpleScaling{T, U}(d₀::T, μ₀::U, σ₀::U, ξ::U, α::U) where {T<:Real, U<:Union{Real, paramfun}} = new{T, U}(d₀, μ₀, σ₀, ξ, α)
end

function SimpleScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T) where {T <: Real}
        
    @assert 0 < α < 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
        
    return SimpleScaling{T, T}(d₀, μ₀, σ₀, ξ, α)
        
end

function SimpleScaling(d₀::T, μ₀::Covariates, σ₀::Covariates, ξ::Covariates, α::Covariates) where {T <: Real}
    μ₀_fun = computeparamfunction(μ₀.parameterization, standardize.(μ₀.covariates))
    σ₀_fun = computeparamfunction(σ₀.parameterization, standardize.(σ₀.covariates))
    ξ_fun = computeparamfunction(ξ.parameterization, standardize.(ξ.covariates))
    α_fun = computeparamfunction(α.parameterization, standardize.(α.covariates))
    return SimpleScaling{T, paramfun}(d₀, paramfun(μ₀.covariates, μ₀_fun, []), paramfun(σ₀.covariates, σ₀_fun, []), paramfun(ξ.covariates, ξ_fun, []), paramfun(α.covariates, α_fun, []))
end

function SimpleScaling(d₀::T, μ₀::paramfun, σ₀::paramfun, ξ::paramfun, α::paramfun) where {T <: Real}
    return SimpleScaling{T, paramfun}(d₀, μ₀, σ₀, ξ, α)
end

SimpleScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real) = SimpleScaling(promote(d₀, μ₀, σ₀, ξ, α)...)

Base.Broadcast.broadcastable(obj::SimpleScaling) = Ref(obj)


### Parameters

"""
    duration(pd::SimpleScaling)

Return the reference duration.
"""
duration(pd::SimpleScaling) = pd.d₀

"""
    exponent(pd::SimpleScaling)

Return the scaling exponent.
"""
exponent(pd::SimpleScaling) = pd.α

location(pd::SimpleScaling) = pd.μ₀

scale(pd::SimpleScaling) = pd.σ₀

shape(pd::SimpleScaling) = pd.ξ

params(pd::SimpleScaling) = (pd.μ₀ isa paramfun ? pd.μ₀.estimators : location(pd), pd.σ₀ isa paramfun ? pd.σ₀.estimators : scale(pd), pd.ξ isa paramfun ? pd.ξ.estimators : shape(pd), pd.α isa paramfun ? pd.α.estimators : exponent(pd))

params_number(::Type{<:SimpleScaling}) = 4

params_number(pd::SimpleScaling) = 4 + (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α isa paramfun ? length(pd.α.covariate) : 0)

function getcovariatenumber(pd::SimpleScaling)::Int
    return sum((pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0) + (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0) + (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0) + (pd.α isa paramfun ? length(pd.α.covariate) : 0))
end


### Methods

"""
    getdistribution(pd::SimpleScaling, d::Real)

Return the marginal GEV distribution for duration `d` according to model pd.
"""
function getdistribution(pd::SimpleScaling, d::Real)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α = (pd.α isa paramfun ? logistic.(pd.α.fun(logit.(pd.α.estimators))) : exponent(pd)) 
    
    d₀ = duration(pd)
    
    ls = -α .* (log.(d) .- log.(d₀))
    s = exp.(ls)

    μ = μ₀ .* s
    σ = σ₀ .* s
    
    return GeneralizedExtremeValue.(μ, σ, ξ)
    
end


"""
    getquantile(pd::SimpleScaling, t::Real, d::Real)

Return the quantile for duration `d` and return level `t`.
"""
function getquantile(pd::SimpleScaling, t::Real, d::Real)
    μ₀ = (pd.μ₀ isa paramfun ? pd.μ₀.fun(pd.μ₀.estimators) : location(pd))
    σ₀ = (pd.σ₀ isa paramfun ? exp.(pd.σ₀.fun(log.(pd.σ₀.estimators))) : scale(pd))
    ξ = (pd.ξ isa paramfun ? pd.ξ.fun(pd.ξ.estimators) : shape(pd)) 
    α = (pd.α isa paramfun ? logistic.(pd.α.fun(logit.(pd.α.estimators))) : exponent(pd))

    p = (1 .- 1 ./ t)
    ls = (-log(p)).^(-ξ)
    
    d₀ = duration(pd)
    
    ls = -α .* (log.(d) .- log.(d₀))
    s = exp.(ls)
    
    return (μ₀ .+ ((σ₀ ./ ξ) .* ((-log(p)).^(-ξ) .- 1))) .* s
end

"""
    construct_model(::Type{<:SimpleScaling}, θ)

Construct a SimpleScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:SimpleScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."

    return SimpleScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]))

end

"""
    construct_model(::Type{<:SimpleScaling}, d₀, θ, c)

Construct a SimpleScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(::Type{<:SimpleScaling}, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."
    return SimpleScaling(d₀, θ_mixed[1], exp(θ_mixed[2]), θ_mixed[3], logistic(θ_mixed[4]))
end

"""
    construct_model(::SimpleScaling, d₀, θ)

Construct a SimpleScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(pd::SimpleScaling, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd)  "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."
    
    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α = (pd.α isa paramfun ? length(pd.α.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ[μ₀_pos : μ₀_pos + cov_μ₀]) : θ[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ[ξ_pos : ξ_pos + cov_ξ]) : θ[ξ_pos]
    α = pd.α isa paramfun ? paramfun(pd.α.covariate, pd.α.fun, logistic.(θ[α_pos : α_pos + cov_α])) : logistic(θ[α_pos])

    return getcovariatenumber(pd) > 0 ? SimpleScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α) : SimpleScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α)
end

"""
    construct_model(::SimpleScaling, d₀, θ, c)

Construct a SimpleScaling marginal model from a set of transformed and fixed parameters in the real space.
"""
function construct_model(pd::SimpleScaling, d₀::Real, θ::AbstractVector{<:Real}, c::AbstractVector{<:Union{Nothing, Real}})
    θ_mixed = [isnothing(fixed_param) ? param : fixed_param for (param, fixed_param) in zip(θ, c)]

    @assert length(θ_mixed) == params_number(pd) "The parameter vector length must be equal to the model parameter number. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α = (pd.α isa paramfun ? length(pd.α.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ

    μ₀ = pd.μ₀ isa paramfun ? paramfun(pd.μ₀.covariate, pd.μ₀.fun, θ_mixed[μ₀_pos : μ₀_pos + cov_μ₀]) : θ_mixed[μ₀_pos]
    σ₀ = pd.σ₀ isa paramfun ? paramfun(pd.σ₀.covariate, pd.σ₀.fun, exp.(θ_mixed[σ₀_pos : σ₀_pos + cov_σ₀])) : exp(θ_mixed[σ₀_pos])
    ξ = pd.ξ isa paramfun ? paramfun(pd.ξ.covariate, pd.ξ.fun, θ_mixed[ξ_pos : ξ_pos + cov_ξ]) : θ_mixed[ξ_pos]
    α = pd.α isa paramfun ? paramfun(pd.α.covariate, pd.α.fun, logistic.(θ_mixed[α_pos : α_pos + cov_α])) : logistic(θ_mixed[α_pos])

    return getcovariatenumber(pd) > 0 ? SimpleScaling{Real, paramfun}(d₀, μ₀, σ₀, ξ, α) : SimpleScaling{Real, Real}(d₀, μ₀, σ₀, ξ, α)
end

"""
    map_to_real_space(::Type{<:SimpleScaling}, θ)

Map the parameters from the SimpleScaling parameter space to the real space.
"""
function map_to_real_space(::Type{<:SimpleScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."

    @assert 0 < θ[4] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[2] > 0 "Scale must be positive"

    return [θ[1], log(θ[2]), θ[3], logit(θ[4])]

end

"""
    map_to_real_space(pd::SimpleScaling, θ)

Map the parameters from the SimpleScaling parameter space to the real hypercube.
"""
function map_to_real_space(pd::SimpleScaling, θ::AbstractVector{<:Real})
    @assert length(θ) == params_number(pd) "The parameter vector length must be 4. Verify that the reference duration is included."

    cov_μ₀ = (pd.μ₀ isa paramfun ? length(pd.μ₀.covariate) : 0)
    cov_σ₀ = (pd.σ₀ isa paramfun ? length(pd.σ₀.covariate) : 0)
    cov_ξ = (pd.ξ isa paramfun ? length(pd.ξ.covariate) : 0)
    cov_α = (pd.α isa paramfun ? length(pd.α.covariate) : 0)

    μ₀_pos = 1
    σ₀_pos = 2 + cov_μ₀
    ξ_pos = 3 + cov_μ₀ + cov_σ₀
    α_pos = 4 + cov_μ₀ + cov_σ₀ + cov_ξ
    
    @assert 0 < θ[α_pos] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[σ₀_pos] > 0 "Scale must be positive"

    μ₀ = cov_μ₀ > 0 ? [θ[μ₀_pos], θ[μ₀_pos + 1 : μ₀_pos + cov_μ₀]...] : θ[μ₀_pos]
    σ₀ = cov_σ₀ > 0 ? log.([θ[σ₀_pos], θ[σ₀_pos + 1 : σ₀_pos + cov_σ₀]...]) : log(θ[σ₀_pos])
    ξ = cov_ξ > 0 ? [θ[ξ_pos], θ[ξ_pos + 1 : ξ_pos + cov_ξ]...] : θ[ξ_pos]
    α = cov_α > 0 ? logit.([θ[α_pos], θ[α_pos + 1 : α_pos + cov_α]...]) : logit(θ[α_pos])

    return [μ₀..., σ₀..., ξ..., α...]

end

"""
    Base.show(io::IO, obj::SimpleScaling)

Override of the show function for the objects of type SimpleScaling.

"""
function Base.show(io::IO, obj::SimpleScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", obj.μ₀ isa paramfun ? round.(obj.μ₀.estimators, digits=4) : round(location(obj), digits=4),
        ", σ₀ = ", obj.σ₀ isa paramfun ? round.(obj.σ₀.estimators, digits=4) : round(scale(obj), digits=4),
        ", ξ = ", obj.ξ isa paramfun ? round.(obj.ξ.estimators, digits=4) : round(shape(obj), digits=4),
        ", α = ", obj.α isa paramfun ? round.(obj.α.estimators, digits=4) : round(exponent(obj), digits=4),
        ")")
end

"""
    initialize(::Type{<:SimpleScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the SimpleScaling marginal model with reference duration d₀, adapted to the data.
The initialization is done by fitting a Gumbel distribution independently for each duration in the data and then estimating the scaling relationship by
    regression over μ and σ. ξ is initialized at 0 as a default.
"""
function initialize(::Type{<:SimpleScaling}, data::IDFdata, d₀::Real)
    
    # step 1 : computing Gumbel parameters separately for each duration
    log_μ_values = Dict{String, Real}()
    log_σ_values = Dict{String, Real}()
    duration_tags = gettag(data)
    for tag in duration_tags
        try
            global fm = Extremes.gevfit(getdata(data, tag))
        catch e 
            global fm = Extremes.gevfitpwm(getdata(data, tag))
        end
        log_μ_values[tag] = log( fm.θ̂[1] )
        log_σ_values[tag] = fm.θ̂[2]
    end

    # step 2 : computing μ_d₀, σ_d₀ et α using regression
    regression_data = DataFrame(is_μ_value = Bool[], is_σ_value = Bool[], log_d = Float64[], param_value = Float64[])
    for tag in duration_tags
        push!(regression_data, [true, false, log(getduration(data, tag) / d₀), log_μ_values[tag]])
        push!(regression_data, [false, true, log(getduration(data, tag) / d₀), log_σ_values[tag]])
    end
    X = Matrix(regression_data[:,1:3])
    y = Vector(regression_data[:,4])
    regression_res = X \ y

    return [ exp(regression_res[1]), 
                maximum([0.001, exp(regression_res[2])]), # avoids possible numerical errors
                0.,
                maximum([0.001, minimum([0.999, - regression_res[3]])]) # avoids possible domain errors
            ]

end

"""
    initialize(::SimpleScaling, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the SimpleScaling marginal model with reference duration d₀, adapted to the data.
The initialization is done by fitting a Gumbel distribution independently for each duration in the data and then estimating the scaling relationship by
    regression over μ and σ. ξ is initialized at 0 as a default.
"""
function initialize(::SimpleScaling, data::IDFdata, d₀::Real)
    
    # step 1 : computing Gumbel parameters separately for each duration
    log_μ_values = Dict{String, Real}()
    log_σ_values = Dict{String, Real}()
    duration_tags = gettag(data)
    for tag in duration_tags
        try
            global fm = Extremes.gevfit(getdata(data, tag))
        catch e 
            global fm = Extremes.gevfitpwm(getdata(data, tag))
        end
        log_μ_values[tag] = log( fm.θ̂[1] )
        log_σ_values[tag] = fm.θ̂[2]
    end

    # step 2 : computing μ_d₀, σ_d₀ et α using regression
    regression_data = DataFrame(is_μ_value = Bool[], is_σ_value = Bool[], log_d = Float64[], param_value = Float64[])
    for tag in duration_tags
        push!(regression_data, [true, false, log(getduration(data, tag) / d₀), log_μ_values[tag]])
        push!(regression_data, [false, true, log(getduration(data, tag) / d₀), log_σ_values[tag]])
    end
    X = Matrix(regression_data[:,1:3])
    y = Vector(regression_data[:,4])
    regression_res = X \ y

    return [ exp(regression_res[1]), 
                maximum([0.001, exp(regression_res[2])]), # avoids possible numerical errors
                0.,
                maximum([0.001, minimum([0.999, - regression_res[3]])]) # avoids possible domain errors
            ]

end