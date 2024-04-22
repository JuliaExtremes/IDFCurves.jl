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
struct GeneralScaling{T<:Real} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::T 
    σ₀::T
    ξ::T
    α::T # duration exponent (defining slope of the IDF curve)
    δ::T # duration offset (defining curvature of the IDF curve)
    GeneralScaling{T}(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T) where {T<:Real} = new{T}(d₀, μ₀, σ₀, ξ, α, δ)
end



function GeneralScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T) where {T <: Real}
        
    @assert 0 < α ≤ 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
    @assert δ ≥ 0 "Duration offset must be non-negative"
        
    return GeneralScaling{T}(d₀, μ₀, σ₀, ξ, α, δ)
        
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

params(pd::GeneralScaling) = (location(pd), scale(pd), shape(pd), exponent(pd), offset(pd))

params_number(::Type{<:GeneralScaling}) = 5

### Methods

"""
    getdistribution(pd::GeneralScaling, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::GeneralScaling, d::Real)
    
    μ₀ = location(pd)
    σ₀ = scale(pd)
    ξ = shape(pd)
    α = exponent(pd)
    δ = offset(pd)
    
    d₀ = duration(pd)
    
    ls = -α * (log(d + δ) - log(d₀ + δ))
    s = exp(ls)

    μ = μ₀ * s
    σ = σ₀ * s
    
    return GeneralizedExtremeValue(μ, σ, ξ)
    
end

"""
    construct_model(::Type{<:GeneralScaling}, d₀, θ)

Construct a GeneralScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:GeneralScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 4. Verify that the reference duration is not included."
    
    return GeneralScaling(d₀, θ[1], exp(θ[2]), logistic(θ[3])-.5, logistic(θ[4]), exp(θ[5]))

end

"""
    map_to_real_space(::Type{<:GeneralScaling}, θ)

Map the parameters from the GeneralScaling parameter spave to the real hypercube.
"""
function map_to_real_space(::Type{<:GeneralScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is not included."

    return [θ[1], log(θ[2]), logit(θ[3]+.5), logit(θ[4]), log(θ[5])]

end

"""
    Base.show(io::IO, obj::GeneralScaling)

Override of the show function for the objects of type GeneralScaling.

"""
function Base.show(io::IO, obj::GeneralScaling)
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", round(location(obj), digits=4),
        ", σ₀ = ", round(scale(obj), digits=4),
        ", ξ = ", round(shape(obj), digits=4),
        ", α = ", round(exponent(obj), digits=4),
        ", δ = ", round(offset(obj), digits=4),
        ")")
end