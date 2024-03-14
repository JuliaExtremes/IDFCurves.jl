struct UncorrelatedStructure <: CorrelationStructure

end

Base.Broadcast.broadcastable(obj::UncorrelatedStructure) = Ref(obj)

params(C::UncorrelatedStructure) = ()

params_number(::Type{<:UncorrelatedStructure}) = 0

function cor(C::UncorrelatedStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."

    return d==0. ? 1 : 0
end

"""
    map_to_param_space(::Type{<:UncorrelatedStructure}, θ)

Map the parameter(s) from the real space to the UncorrelatedStructure parameter space.
"""
function map_to_param_space(::Type{<:UncorrelatedStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 0 "The parameter vector length must be 0 for an uncorrelated structure."

    return Float64[]

end

"""
    map_to_real_space(::Type{<:UncorrelatedStructure}, θ)

Map the parameter(s) from the UncorrelatedStructure parameter space to the real space.
"""
function map_to_real_space(::Type{<:UncorrelatedStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 0 "The parameter vector length must be 0 for an uncorrelated structure."

    return Float64[]

end