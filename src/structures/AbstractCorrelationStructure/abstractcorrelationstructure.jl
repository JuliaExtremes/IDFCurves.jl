
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



struct MaternCorrelationStructure <: AbstractCorrelationStructure
    ν::Float64
    ρ::Float64

    function MaternCorrelationStructure(ν::Real, ρ::Real)
        @assert ν > 0 "Matern correlogram parameter ν must be positive"   
        @assert ρ >0 "Matern correlogram parameter ρ must be positive"      
        return new(float(ν), float(ρ))
    end
end

Base.Broadcast.broadcastable(obj::MaternCorrelationStructure) = Ref(obj)

params(C::MaternCorrelationStructure) = (C.ν, C.ρ)

function cor(C::MaternCorrelationStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."

    if d ≈ 0
        c = 1.
    else
        ν, ρ = params(C)

        z = sqrt(2*ν)*d/ρ

        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * BesselK.adbesselkxv(ν, z)  # the term z^v seems already included in BesselK.adbesselkxv

    end

    return c
end











    # function matern(x, y, params)
    #     (sg, rho, nu) = (params[1], params[2], params[3])
    #     dist = _norm(x-y)
    #     _iszero(dist) && return sg^2
    #     arg = sqrt(2*nu)*dist/rho
    #     (sg*sg*(2^(1-nu))/_gamma(nu))*adbesselkxv(nu, arg)
    #   end