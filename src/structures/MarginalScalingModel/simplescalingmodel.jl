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
struct SimpleScaling{T<:Real} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::T 
    σ₀::T
    ξ::T
    α::T # scaling exponent (defining slope of the IDF curve)
    SimpleScaling{T}(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T) where {T<:Real} = new{T}(d₀, μ₀, σ₀, ξ, α)
end

function SimpleScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T) where {T <: Real}
        
    @assert 0 < α ≤ 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
        
    return SimpleScaling{T}(d₀, μ₀, σ₀, ξ, α)
        
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

params(pd::SimpleScaling) = (location(pd), scale(pd), shape(pd), exponent(pd))

params_number(::Type{<:SimpleScaling}) = 4


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
    construct_model(::Type{<:SimpleScaling}, θ)

Construct a SimpleScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:SimpleScaling}, d₀::Real, θ::AbstractVector{<:Real})
    @assert length(θ) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."

    return SimpleScaling(d₀, θ[1], exp(θ[2]), logistic(θ[3])-.5, logistic(θ[4]))

end

"""
    map_to_real_space(::Type{<:SimpleScaling}, θ)

Map the parameters from the SimpleScaling parameter space to the real space.
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
    println(io, 
        typeof(obj), "(",
        "d₀ = ", duration(obj),
        ", μ₀ = ", round(location(obj), digits=4),
        ", σ₀ = ", round(scale(obj), digits=4),
        ", ξ = ", round(shape(obj), digits=4),
        ", α = ", round(exponent(obj), digits=4),
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
        fm = Extremes.gevfit(getdata(data, tag))
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
    [exp(regression_res[1]), exp(regression_res[2]), - regression_res[3]]

    return [ exp(regression_res[1]), exp(regression_res[2]), 0.,  - regression_res[3] ]

end