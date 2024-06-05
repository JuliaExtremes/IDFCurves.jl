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
    construct_model(::Type{<:ExponentialCorrelationStructure, θ)

Construct an ExponentialCorrelationStructure from a set of transformed parameters θ in the real space.
"""
function construct_model(::Type{<:ExponentialCorrelationStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 1 "The parameter vector length must be 1 for an exponential correlation structure."

    return ExponentialCorrelationStructure(exp(θ[1]))

end

"""
    map_to_real_space(::Type{<:ExponentialCorrelationStructure}, θ)

Map the parameter(s) from the ExponentialCorrelationStructure parameter space to the real space.
"""
function map_to_real_space(::Type{<:ExponentialCorrelationStructure}, θ::AbstractVector{<:Real})
    @assert length(θ) == 1 "The parameter vector length must be 1 for an exponential correlation structure."

    @assert θ[1] > 0 "exponential correlogram parameter must be positive"  

    return [log(θ[1])]

end

"""
    initialize(::Type{<:ExponentialCorrelationStructure}, data::IDFdata)

Initialize a vector of parameters for the ExponentialCorrelationStructure adapted to the data.
The initialization is done by fitting the correlation function to the Kendall's Tau (measure of correlation) associated to each pair of durations.
"""
function initialize(::Type{<:ExponentialCorrelationStructure}, data::IDFdata)

    # Kendall's Tau for each pair of durations
    kendall_data = IDFCurves.getKendalldata(data)
    transform!(kendall_data, :kendall => (x -> sin.(pi / 2 .* x)) => :kendall)

    # The function to be optimized takes as argument a vector of size 1 containing the value (transformed into real space) of the correlation parameter, 
    # and returns the squared error associated with the approximation of the empirical Kendall's Tau by the theoretical exponential correlation with this parameter.
    function MSE_kendall(θ::DenseVector{<:Real})
        cor_struct =  construct_model(ExponentialCorrelationStructure, θ)
        corrs = [ cor(cor_struct, h) for h in kendall_data[:,:distance] ]
    
        return sum( (corrs .- kendall_data[:,:kendall]).^2 )
    end

    # optimization
    θ₀ = map_to_real_space(ExponentialCorrelationStructure, [1.])
    θ̂ = perform_optimization(MSE_kendall, θ₀, warn_message = "Automatic initialization did not work as expected for the ExponentialCorrelationStructure. Initialized parameter is 1 as a default.")

    return [params(construct_model(ExponentialCorrelationStructure, θ̂))...]

end