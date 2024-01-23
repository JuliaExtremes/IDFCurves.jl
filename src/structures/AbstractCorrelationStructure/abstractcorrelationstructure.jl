
abstract type AbstractCorrelationStructure end

struct ExponentialCorrelationStructure <: AbstractCorrelationStructure
    θ::Float64
    function ExponentialCorrelationStructure(θ::Real)
        @assert θ > 0 "exponential correlogram parameter must be positive"        
        return new(float(θ))
    end
end

Base.Broadcast.broadcastable(obj::ExponentialCorrelationStructure) = Ref(obj)

params(C::ExponentialCorrelationStructure) = (C.θ)

function cor(C::ExponentialCorrelationStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."

    ρ = params(C)[]

    return exp(-d/ρ)
end



# struct MaternCorrelationStructure <: AbstractCorrelationStructure
#     ν::Real
#     ρ::Real

#     function MaternCorrelationStructure(ν::Real, ρ::Real)
#         @assert ν > 0 "Matern correlogram parameter ν must be positive"   
#         @assert ρ >0 "Matern correlogram parameter ρ must be positive"      
#         return new(float(ν), float(ρ))
#     end
# end


struct MaternCorrelationStructure{T<:Real} <: AbstractCorrelationStructure
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

    if d ≈ 0
        c = 1.
    else
        ν, ρ = params(C)

        z = sqrt(2*ν)*d/ρ

        # c = 2^(1-ν)/SpecialFunctions.gamma(ν) *z^ν* Bessels.besselk(ν, z)  # with Bessels.jl (not compatible with automatic differentiation)
        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * BesselK.adbesselkxv(ν, z)  # with Bessels.jl (not compatible with automatic differentiation), the term z^ν is included in adbesselkxv

    end

    return c
end


function cor(C::AbstractCorrelationStructure, h::AbstractMatrix{<:Real})
    @assert issymmetric(h)

    M = Matrix{Float64}(I, size(h))

    for i in 1:size(h,1)
        for j in 1:size(h,2)
            (i ≤ j) ? M[i,j] = cor(C, h[i,j]) : continue
        end
    end

    return PDMat(Symmetric(M))

end










    # function matern(x, y, params)
    #     (sg, rho, nu) = (params[1], params[2], params[3])
    #     dist = _norm(x-y)
    #     _iszero(dist) && return sg^2
    #     arg = sqrt(2*nu)*dist/rho
    #     (sg*sg*(2^(1-nu))/_gamma(nu))*adbesselkxv(nu, arg)
    #   end