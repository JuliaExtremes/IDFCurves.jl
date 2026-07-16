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

function SimpleScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α::T) where {T<:Real}

    @assert 0 < α < 1 "Scaling exponent must be between 0 and 1"
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
    scaling_factor(sm::SimpleScaling, d::Real)

Compute the scaling factor for duration `d` under the Simple Scaling model `sm`.

### Details

The scaling factor is defined as

```math
s(d) = \frac{(d)^{-\\alpha}}{(d_0)^{-\\alpha}}.
```
"""
function scaling_factor(sm::SimpleScaling, d::Real)

    @assert d > 0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    α = exponent(sm)

    s = exp(-α * log1p((d - d₀) / d₀))

    return s

end

"""
    construct_model(::Type{<:SimpleScaling}, θ)

Construct a SimpleScaling marginal model from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:SimpleScaling}, d₀::Real, θ::AbstractVector{<:Real};
    final_model::Bool=false)
    @assert length(θ) == 4 "The parameter vector length must be 4. Verify that the reference duration is not included."

    return SimpleScaling(d₀, θ[1], exp(θ[2]), θ[3], logistic(θ[4]))

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
    initialize(::Type{<:SimpleScaling}, data::IDFdata, d₀::Real, lower_threshold::Real=0)

Construct an initial `SimpleScaling` model from IDF data.

Gumbel distributions are fitted independently at each duration using probability
weighted moments. The scaling exponent and reference-location parameter are then
initialized by a log-log regression of the fitted Gumbel locations on duration.
The reference-scale parameter is initialized from the fitted Gumbel scales using
the same scaling exponent.
"""
function initialize(::Type{<:SimpleScaling}, data::IDFdata, d₀::Real, lower_threshold::Real=0.0)

    d₀ > 0 || throw(ArgumentError("Reference duration must be positive, got d₀=$d₀"))

    d = Float64.(getduration.(data, gettag(data)))
    filter!(≥(lower_threshold), d)

    length(d) ≥ 2 || throw(ArgumentError("Lower threshold is too high, at least two durations are required to initialize SimpleScaling."))

    tags = gettag.(data, d)

    μ = Vector{Float64}(undef, length(tags))
    σ = Vector{Float64}(undef, length(tags))

    for (i, tag) in enumerate(tags)
        fd = fit(Gumbel, getdata(data, tag), method="pwm")

        μ[i] = location(fd)
        σ[i] = Distributions.scale(fd)
    end

    all(>(0), μ) || throw(ArgumentError("Fitted Gumbel locations must be positive to initialize SimpleScaling."))
    all(>(0), σ) || throw(ArgumentError("Fitted Gumbel scales must be positive to initialize SimpleScaling."))

    logd = log.(d ./ d₀)

    X = [ones(length(tags)) logd]
    β = X \ log.(μ)

    α = clamp(-β[2], 0.001, 0.999)

    μ₀ = exp(β[1])
    σ₀ = exp(mean(log.(σ) .+ α .* logd))

    return SimpleScaling(d₀, μ₀, σ₀, 0.0, α)

end