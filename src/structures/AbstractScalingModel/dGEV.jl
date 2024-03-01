"""
    dGEV(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real)

Construct a dGEV distribution type.

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
struct dGEV{T<:Real} <: AbstractScalingModel
    d₀::T # reference duration
    μ₀::T 
    σ₀::T
    ξ::T
    α::T # duration exponent (defining slope of the IDF curve)
    δ::T # duration offset (defining curvature of the IDF curve)
    dGEV{T}(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T) where {T<:Real} = new{T}(d₀, μ₀, σ₀, ξ, α, δ)
end



function dGEV(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T, δ::T) where {T <: Real}
        
    @assert 0 < α ≤ 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
    @assert δ ≥ 0 "Duration offset must be non-negative"
        
    return dGEV{T}(d₀, μ₀, σ₀, ξ, α, δ)
        
end

dGEV(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real) = dGEV(promote(d₀, μ₀, σ₀, ξ, α, δ)...)

Base.Broadcast.broadcastable(obj::dGEV) = Ref(obj)


### Parameters

"""
    duration(pd::dGEV)

Return the reference duration.
"""
duration(pd::dGEV) = pd.d₀

"""
    exponent(pd::dGEV)

Return the duration exponent.
"""
exponent(pd::dGEV) = pd.α

location(pd::dGEV) = pd.μ₀

"""
    offset(pd::dGEV)

Return the duration offset
"""
offset(pd::dGEV) = pd.δ

params(pd::dGEV) = (location(pd), scale(pd), shape(pd), exponent(pd), offset(pd))

scale(pd::dGEV) = pd.σ₀

shape(pd::dGEV) = pd.ξ



### Methods

"""
    getdistribution(pd::dGEV, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::dGEV, d::Real)
    
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
    map_to_param_space(::Type{<:dGEV}, θ)

Map the parameters from the real hypercube to the dGEV parameter space.
"""
function map_to_param_space(::Type{<:dGEV}, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is not included."

    return [θ[1], exp(θ[2]), logistic(θ[3])-.5, logistic(θ[4]), exp(θ[5])]

end

"""
    map_to_real_space(::Type{<:dGEV}, θ)

Map the parameters from the dGEV parameter spave to the real hypercube.
"""
function map_to_real_space(::Type{<:dGEV}, θ::AbstractVector{<:Real})
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is not included."

    return [θ[1], log(θ[2]), logit(θ[3]+.5), logit(θ[4]), log(θ[5])]

end

"""
    Base.show(io::IO, obj::dGEV)

Override of the show function for the objects of type dGEV.

"""
function Base.show(io::IO, obj::dGEV)
    println(io, 
        typeof(obj), "(",
        "μ₀ = ", round(location(obj), digits=4),
        ", σ₀ = ", round(scale(obj), digits=4),
        ", ξ = ", round(shape(obj), digits=4),
        ", α = ", round(exponent(obj), digits=4),
        ", δ = ", round(offset(obj), digits=4),
        ")")
end