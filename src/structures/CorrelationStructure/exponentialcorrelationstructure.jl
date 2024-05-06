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

    return [log(θ[1])]

end

"""
    initialize(::Type{<:ExponentialCorrelationStructure}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the ExponentialCorrelationStructure adapted to the data.
The initialization is done by fitting the correlation function to the Kendall's Tau (measure of correlation) associated to each pair of durations.
"""
function initialize(::Type{<:ExponentialCorrelationStructure}, data::IDFdata)

    # Kendall's Tau for each pair of durations
    kendall_data = IDFCurves.getKendalldata(data)

    # The function to be optimized takes as argument a vector of size 1 containing the value (transformed into real space) of the correlation parameter, 
    # and returns the squared error associated with the approximation of the empirical Kendall's Tau by the theoretical exponential correlation with this parameter.
    function MSE_kendall(θ)
        cor_struct =  construct_model(ExponentialCorrelationStructure, θ)
        corrs = [ cor(cor_struct, h) for h in kendall_data[:,:distance] ]
    
        return sum( (corrs .- kendall_data[:,:kendall]).^2 )
    end

    # associated gradient and hessian
    function grad_MSE_kendall(G, θ)
        grad = ForwardDiff.gradient(MSE_kendall, θ)
        for i in eachindex(G)
            G[i] = grad[i]
        end
    end
    function hessian_MSE_kendall(H, θ)
        hess = ForwardDiff.hessian(MSE_kendall, θ)
        for i in axes(H,1)
            for j in axes(H,2)
                H[i,j] = hess[i,j]
            end
        end
    end

    # optimization
    res = nothing
    try 
        res = Optim.optimize(MSE_kendall, grad_MSE_kendall, hessian_MSE_kendall, map_to_real_space(ExponentialCorrelationStructure, [1.]))
        @assert Optim.converged(res)
    catch e
        res = Optim.optimize(MSE_kendall, map_to_real_space(ExponentialCorrelationStructure, [1.]))
    end

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
        init_params = [params(construct_model(ExponentialCorrelationStructure, θ̂))...]
    else
        @warn "Automatic initialization did not work as expected for the ExponentialCorrelationStructure. Initialized parameter is 1 as a default."
        init_params = [1.]
    end

    return init_params

end