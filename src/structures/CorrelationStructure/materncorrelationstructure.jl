struct MaternCorrelationStructure{T<:Real} <: CorrelationStructure
    ν::T
    ρ::T
    MaternCorrelationStructure{T}(ν::T, ρ::T) where {T<:Real} = new{T}(ν, ρ)
end


function MaternCorrelationStructure(ν::T, ρ::T) where {T <: Real}
    @assert ν > 0 "Matern correlogram parameter ν must be positive"   
    @assert ρ >0 "Matern correlogram parameter ρ must be positive"    
    return MaternCorrelationStructure{T}(ν, ρ)
end

MaternCorrelationStructure(ν::Real, ρ::Real) = MaternCorrelationStructure(promote(ν, ρ)...)
MaternCorrelationStructure(ν::Integer, ρ::Integer) = MaternCorrelationStructure(float(ν), float(ρ))

Base.Broadcast.broadcastable(obj::MaternCorrelationStructure) = Ref(obj)

params(C::MaternCorrelationStructure) = (C.ν, C.ρ)

function cor(C::MaternCorrelationStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."
    
        ν, ρ = params(C)
        z = sqrt(2*ν)*d/ρ

        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * BesselK.adbesselkxv(ν, z)

    return c
end