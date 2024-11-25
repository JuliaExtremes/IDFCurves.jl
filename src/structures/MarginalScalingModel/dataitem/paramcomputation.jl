abstract type ParamComputation end

struct BaseParamComputation <: ParamComputation end
struct LinearParamComputation <: ParamComputation end
struct NonDimLinearParamComputation <: ParamComputation end
struct NonDimExpoParamComputation <: ParamComputation end

function computeparamfunction(::ParamComputation, covariates::Vector{<:DataItem})::Function
    throw(MethodError("computeparamfunction must be implemented for the specific ParamComputation subtype"))
end

function computeparamfunction(::Type{BaseParamComputation}, covariates::Vector{<:DataItem})::Function
    fun =
    if isempty(covariates)
        function(β::Vector{<:Real})
            return identity(β)
        end
    else
        X = ones(length(covariates[1].value))

        for cov in covariates
            X = hcat(X, cov.value)
        end
        function(β::Vector{<:Real})
            return X*β
        end
    end
    return fun

end

"""
    (covariates::Vector{Covariate})

Establish the parameter as function of the corresponding covariates
for the form `μ = μ0 + (μ1 * x)`.
"""
function computeparamfunction(::Type{LinearParamComputation}, covariates::Vector{<:DataItem})::Function
    if isempty(covariates)
        return β -> identity(β)
    else
        X = hcat(ones(length(covariates[1].value)), [cov.value for cov in covariates]...)
        return β -> X * β
    end
end

"""
    (covariates::Vector{Covariate})

Establish the parameter as a function of the corresponding covariates in a non-dimensionalized form
for the form `μ = μ0 * (1 + n * x)`.
"""
function computeparamfunction(::Type{NonDimLinearParamComputation}, covariates::Vector{<:DataItem})::Function
    if isempty(covariates)
        return β -> identity(β[1])  # Only μ0 is used
    else
        X = hcat(ones(length(covariates[1].value)), [cov.value for cov in covariates]...)
        return β -> β[1] * (1 .+ X[:, 2:end] * β[2:end])
    end
end


"""
    computeparamfunction(covariates::Vector{DataItem})

Establish the parameter as a function of the corresponding covariates in a non-dimensionalized form
for the form `μ = μ0 * exp(n * x)`.
"""
function computeparamfunction(::Type{NonDimExpoParamComputation}, covariates::Vector{<:DataItem})::Function
    if isempty(covariates)
        return β -> identity(β[1])  # Only μ0 is used
    else
        X = hcat([cov.value for cov in covariates]...)
        return β -> β[1] .* exp.(X * β[2:end])
    end
end
