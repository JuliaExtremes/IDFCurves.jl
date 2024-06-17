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

params_number(::Type{<:MaternCorrelationStructure}) = 2

function cor(C::MaternCorrelationStructure, d::Real)
    @assert d ≥ 0 "distance must be non-negative."
    
        ν, ρ = params(C)
        z = sqrt(2*ν)*d/ρ

        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * BesselK.adbesselkxv(ν, z)

    return c
end

"""
    construct_model(::Type{<:MaternCorrelationStructure, θ)

Construct an MaternCorrelationStructure from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:MaternCorrelationStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 2 "The parameter vector length must be 1 for a Matern correlation structure."

    return MaternCorrelationStructure(exp(θ[1]), exp(θ[2]))

end

"""
    map_to_real_space(::Type{<:MaternCorrelationStructure}, θ)

Map the parameter(s) from the MaternCorrelationStructure parameter space to the real space.
"""
function map_to_real_space(::Type{<:MaternCorrelationStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 2 "The parameter vector length must be 1 for an exponential correlation structure."

    @assert θ[1] > 0 "Matern correlogram parameter ν must be positive"   
    @assert θ[2] >0 "Matern correlogram parameter ρ must be positive"

    return [log(θ[1]), log(θ[2])]

end

"""
    initialize(::Type{<:MaternCorrelationStructure}, data::IDFdata)

Initialize a vector of parameters for the MaternCorrelationStructure adapted to the data.
The initialization is done by fitting the correlation function to the Kendall's Tau (measure of correlation) associated to each pair of durations.
"""
function initialize(::Type{<:MaternCorrelationStructure}, data::IDFdata)

    # Kendall's Tau for each pair of durations
    kendall_data = IDFCurves.getKendalldata(data)
    transform!(kendall_data, :kendall => (x -> sin.(pi / 2 .* x)) => :kendall)

    # The function to be optimized takes as argument a vector of size 2 containing the values (transformed into real space) of the correlation parameters, 
    # and returns the squared error associated with the approximation of the empirical Kendall's Tau by the theoretical exponential correlation with these parameters.
    function MSE_kendall(θ::DenseVector{<:Real})
        cor_struct =  construct_model(MaternCorrelationStructure, θ)
        corrs = [ cor(cor_struct, h) for h in kendall_data[:,:distance] ]
    
        return sum( (corrs .- kendall_data[:,:kendall]).^2 )
    end

    # optimization
    θ₀ = [0., 0.]
    θ̂ = perform_optimization(MSE_kendall, θ₀, warn_message = "Automatic initialization did not work as expected for the MaternCorrelationStructure. Initialized parameters are (1,1) as a default.")

    return [maximum([0.001, exp(θ̂[1])]), # avoids possible numerical errors
            maximum([0.001, exp(θ̂[2])]) # avoids possible numerical errors
            ]
end