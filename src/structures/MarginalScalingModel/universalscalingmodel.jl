"""
    UniversalScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real, τ::Real)

Construct a UniversalScaling distribution type.

## Details

**TODO**

``\\mu_d = \\mu_0 \\left(\\frac{(d+\\delta)^{-\\alpha}+\\tau}{(d_0+\\delta)^{-\\alpha}+\\tau} \\right) \\qquad 
  \\sigma_d = \\sigma_0 \\left(\\frac{(d+\\delta)^{-\\alpha}+\\tau}{(d_0+\\delta)^{-\\alpha}+\\tau} \\right) \\qquad 
  \\xi_d = \\xi``

"""
struct UniversalScaling{T<:Real} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::T 
    σ₀::T
    ξ::T
    α::T # duration exponent (defining slope of the log-log IDF curve)
    δ::T # duration offset (defining concavity of the log-log IDF curve)
    τ::T # large-scale offset (defining convexity of the log-log IDF curve)
    UniversalScaling{T}(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T, τ::T) where {T<:Real} = new{T}(d₀, μ₀, σ₀, ξ, α, δ, τ)
end


function UniversalScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T, τ::T) where {T <: Real}
        
    @assert 0 < α < 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
    @assert δ ≥ 0 "Duration offset must be non-negative"
    @assert τ ≥ 0 "Large-scale offset must be non-negative"
        
    return UniversalScaling{T}(d₀, μ₀, σ₀, ξ, α, δ, τ)
        
end

UniversalScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real, τ::Real) = UniversalScaling(promote(d₀, μ₀, σ₀, ξ, α, δ, τ)...)

Base.Broadcast.broadcastable(obj::UniversalScaling) = Ref(obj)


### Parameters

"""
    duration(pd::UniversalScaling)

Return the reference duration.
"""
duration(pd::UniversalScaling) = pd.d₀

"""
    exponent(pd::UniversalScaling)

Return the duration exponent.
"""
exponent(pd::UniversalScaling) = pd.α

location(pd::UniversalScaling) = pd.μ₀

"""
    offset(pd::UniversalScaling)

Return the duration offset
"""
offset(pd::UniversalScaling) = pd.δ

scale(pd::UniversalScaling) = pd.σ₀

shape(pd::UniversalScaling) = pd.ξ

"""
    largescale_offset(pd::UniversalScaling)

Return the large-scale offset τ
"""
largescale_offset(pd::UniversalScaling) = pd.τ



params(pd::UniversalScaling) = (location(pd), scale(pd), shape(pd), exponent(pd), offset(pd), largescale_offset(pd))

params_number(::Type{<:UniversalScaling}) = 6

### Methods

"""
    getdistribution(pd::UniversalScaling, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::UniversalScaling, d::Real)
    

    d₀ = duration(pd) 
    μ₀, σ₀, ξ, α, δ, τ = params(pd)


    # μ₀ = location(pd)
    # σ₀ = scale(pd)
    # ξ = shape(pd)
    # α = exponent(pd)
    # δ = offset(pd)
    # τ = largescale_offset(pd)
    
    ls = log((d + δ)^-α + τ) - log((d₀ + δ)^-α + τ)
    s = exp(ls)

    μ = μ₀ * s
    σ = σ₀ * s

    return GeneralizedExtremeValue(μ, σ, ξ)
    
end

"""
    construct_model(::Type{<:UniversalScaling}, d₀, θ)

Construct a UniversalScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:UniversalScaling}, d₀::Real, θ::AbstractVector{<:Real};
                            final_model::Bool = false)
    @assert length(θ) == 6 "The parameter vector length must be 6. Verify that the reference duration is not included."

    if final_model && exp(θ[5]) <= 1e-8
        @warn "The value for δ is smaller than 1e-8 so it is set to 0."
        return UniversalScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), 0., exp(θ[6]))

    elseif final_model && exp(θ[6]) <= 1e-8
        @warn "The value for τ is smaller than 1e-8 so it is set to 0."
        return UniversalScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), exp(θ[5]), 0.)

    elseif final_model && exp(θ[5]) <= 1e-8 && exp(θ[6]) <= 1e-8
        @warn "The values for δ and τ are smaller than 1e-8 so they are set to 0."
        return UniversalScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), 0., 0.)

    else
        return UniversalScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]), exp(θ[5]), exp(θ[6]))
    end

end

"""
    map_to_real_space(::Type{<:UniversalScaling}, θ)

Map the parameters from the UniversalScaling parameter space to the real hypercube.
"""
function map_to_real_space(::Type{<:UniversalScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 6 "The parameter vector length must be 6. Verify that the reference duration is not included."

    @assert 0 < θ[4] < 1 "Scaling exponent must be between 0 and 1"
    @assert θ[2] > 0 "Scale must be positive"
    @assert θ[5] ≥ 0 "Duration offset must be non-negative"
    @assert θ[6] ≥ 0 "Minimum intensity must be non-negative"

    return [θ[1], log(θ[2]), θ[3], logit(θ[4]), log(θ[5]), log(θ[6])]

end

"""
    Base.show(io::IO, obj::UniversalScaling)

Override of the show function for the objects of type UniversalScaling.

"""
function Base.show(io::IO, obj::UniversalScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", round(location(obj), digits=4),
        ", σ₀ = ", round(scale(obj), digits=4),
        ", ξ = ", round(shape(obj), digits=4),
        ", α = ", round(exponent(obj), digits=4),
        ", δ = ", round(offset(obj), digits=4),
        ", τ = ", round(largescale_offset(obj), digits=4),
        ")")
end

"""
    initialize(::Type{<:UniversalScaling}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the UniversalScaling marginal model with reference duration d₀, adapted to the data.
The initialization is the same as for the GeneralScaling model. τ is initialized at (close to) 0 as a default.
"""
function initialize(::Type{<:UniversalScaling}, data::IDFdata, d₀::Real)
    
    pd = IDFCurves.fit_mle(GeneralScaling, data, d₀)
    init_general_scaling = [pd.μ₀, pd.σ₀, pd.ξ, pd.α, pd.δ]
    return vcat(init_general_scaling, 0.01)

end