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
struct SimpleScaling <: AbstractScalingModel
    d₀::Real # reference duration
    μ₀::Real 
    σ₀::Real
    ξ::Real
    α::Real # scaling exponent (defining slope of the IDF curve)

    function SimpleScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real)
        
        @assert 0 < α ≤ 1 "Scaling exponent must be between 0 and 1"
        @assert σ₀ > 0 "Scale must be positive"
        
        return new(d₀, μ₀, σ₀, ξ, α)
        
    end
end

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

params(pd::SimpleScaling) = (location(pd), scale(pd), shape(pd), exponent(pd))

scale(pd::SimpleScaling) = pd.σ₀

shape(pd::SimpleScaling) = pd.ξ



### Methods

"""
    getdistribution(pd::SimpleScaling, d::Real)

Return the marginal GEV distribution for duration `d` according to model pd.
"""
function getdistribution(pd::SimpleScaling, d::Real)
    
    μ₀ = location(pd)
    σ₀ = scale(pd)
    ξ = shape(pd)
    α = exponent(pd)
    
    d₀ = duration(pd)
    
    ls = -α * (log(d) - log(d₀))
    s = exp(ls)

    μ = μ₀ * s
    σ = σ₀ * s
    
    return GeneralizedExtremeValue(μ, σ, ξ)
    
end

"""
    map_to_param_space(::Type{<:SimpleScaling}, θ)

Map the parameters from the real hypercube to the SimpleScaling parameter space.
"""
function map_to_param_space(::Type{<:SimpleScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."

    return [θ[1], exp(θ[2]), logistic(θ[3])-.5, logistic(θ[4])]

end

"""
    map_to_real_space(::Type{<:SimpleScaling}, θ)

Map the parameters from the SimpleScaling parameter spave to the real hypercube.
"""
function map_to_real_space(::Type{<:SimpleScaling}, θ::AbstractVector{<:Real})
    @assert length(θ) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."

    return [θ[1], log(θ[2]), logit(θ[3]+.5), logit(θ[4])]

end

"""
    Base.show(io::IO, obj::SimpleScaling)

Override of the show function for the objects of type SimpleScaling.

"""
function Base.show(io::IO, obj::SimpleScaling)
    println(io, "SimpleScaling(d₀ = ", duration(obj),
        ", μ₀ = ", round(location(obj), digits=4),
        ", σ₀ = ", round(scale(obj), digits=4),
        ", ξ = ", round(shape(obj), digits=4),
        ", α = ", round(exponent(obj), digits=4),
        ")")
end