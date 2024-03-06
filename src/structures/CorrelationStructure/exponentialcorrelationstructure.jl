struct ExponentialCorrelationStructure{T<:Real} <: CorrelationStructure
    θ::T
    function ExponentialCorrelationStructure(θ::T) where {T<:Real}
        @assert θ > 0 "exponential correlogram parameter must be positive"     
        return new{T}(θ)
    end
end

Base.Broadcast.broadcastable(obj::ExponentialCorrelationStructure) = Ref(obj)

params(C::ExponentialCorrelationStructure) = (C.θ)

function cor(C::ExponentialCorrelationStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."

    ρ = params(C)[]

    return exp(-d/ρ)
end