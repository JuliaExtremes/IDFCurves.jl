# using Pkg
# pkg"activate ."

# using CSV, DataFrames, Distributions, IDFCurves, LinearAlgebra, Random


import IDFCurves: duration, exponent, location, scale, shape, params, params_number, scaling_factor, getdistribution

# Hybrid scaling Model

struct HybridScaling{T<:Real} <: MarginalScalingModel
    d₀::T # reference duration
    μ₀::T 
    σ₀::T
    ξ::T
    α₁::T # exponent before d₀ (defining slope of the IDF curve)
    α₂::T # exponent after d₀ (defining slope of the IDF curve)
    HybridScaling{T}(d₀::T, μ₀::T, σ₀::T, ξ::T, α₁::T, α₂::T) where {T<:Real} = new{T}(d₀, μ₀, σ₀, ξ, α₁, α₂)
end


function HybridScaling(d₀::T, μ₀::T, σ₀::T, ξ::T, α₁::T, α₂::T) where {T <: Real}
        
    @assert 0 < α₁ < 1 "Scaling exponent must be between 0 and 1"
    @assert 0 < α₂ < 1 "Scaling exponent must be between 0 and 1"
    @assert σ₀ > 0 "Scale must be positive"
        
    return HybridScaling{T}(d₀, μ₀, σ₀, ξ, α₁, α₂)
        
end

HybridScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α₁::Real, α₂::Real) = HybridScaling(promote(d₀, μ₀, σ₀, ξ, α₁, α₂)...)

Base.Broadcast.broadcastable(obj::HybridScaling) = Ref(obj)

### Parameters


duration(pd::HybridScaling) = pd.d₀
exponent(pd::HybridScaling) = (pd.α₁, pd.α₂)
location(pd::HybridScaling) = pd.μ₀
scale(pd::HybridScaling) = pd.σ₀
shape(pd::HybridScaling) = pd.ξ
params(pd::HybridScaling) = (location(pd), scale(pd), shape(pd), exponent(pd)...)
params_number(::Type{<:HybridScaling}) = 5

### Methods

function scaling_factor(sm::HybridScaling, d::Real)

    @assert d>0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    if d < d₀
        α = first(exponent(sm))
    else
        α = last(exponent(sm))
    end

    s = exp(-α * log1p((d - d₀) / d₀))

    return s

end


# model = HybridScaling(1., 20, 5, 0.1, .5, .7)

# params(model)

# scaling_factor(model, .5)

# IDFCurves.getdistribution(model, 1.)


# tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
# durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
# duration_dict = Dict(zip(tags, durations))

