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
    construct_model(::Type{<:MaternCorrelationStructure, θ)

Construct an UncorrelatedStructure from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:UncorrelatedStructure}, θ::AbstractVector{<:Any})
    @assert length(θ) == 0 "The parameter vector length must be 0 for an uncorrelated structure."

    return UncorrelatedStructure()

end

"""
    map_to_real_space(::Type{<:UncorrelatedStructure}, θ)

Map the parameter(s) from the UncorrelatedStructure parameter space to the real space.
"""
function map_to_real_space(::Type{<:UncorrelatedStructure}, θ::AbstractVector{<:Any})
    @assert length(θ) == 0 "The parameter vector length must be 0 for an uncorrelated structure."

    return Float64[]

end

"""
    initialize(::Type{<:UncorrelatedStructure}, data::IDFdata)

Returs an empty vector.
"""
function initialize(::Type{<:UncorrelatedStructure}, data::IDFdata)

    return Float64[]

end