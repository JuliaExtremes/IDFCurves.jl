struct ExponentialCorrelationStructure{T<:Real} <: CorrelationStructure
    θ::T
    function ExponentialCorrelationStructure(θ::T) where {T<:Real}
        @assert θ > 0 "exponential correlogram parameter must be positive"     
        return new{T}(θ)
    end
end

Base.Broadcast.broadcastable(obj::ExponentialCorrelationStructure) = Ref(obj)

params(C::ExponentialCorrelationStructure) = (C.θ)

params_number(::Type{<:ExponentialCorrelationStructure}) = 1

function cor(C::ExponentialCorrelationStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."

    ρ = params(C)[]

    return exp(-d/ρ)
end


"""
    map_to_param_space(::Type{<:ExponentialCorrelationStructure}, θ)

Map the parameter(s) from the real space to the ExponentialCorrelationStructure parameter space.
"""
function map_to_param_space(::Type{<:ExponentialCorrelationStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 1 "The parameter vector length must be 1 for an exponential correlation structure."

    return [exp(θ[1])]

end

"""
    map_to_real_space(::Type{<:ExponentialCorrelationStructure}, θ)

Map the parameter(s) from the ExponentialCorrelationStructure parameter space to the real space.
"""
function map_to_real_space(::Type{<:ExponentialCorrelationStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 1 "The parameter vector length must be 1 for an exponential correlation structure."

    return [log(θ[1])]

end