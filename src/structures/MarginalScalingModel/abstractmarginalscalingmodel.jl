abstract type MarginalScalingModel <: ContinuousMultivariateDistribution end

include("simplescalingmodel.jl")
include("generalscalingmodel.jl")


### Methods

"""
    cdf(pd::MarginalScalingModel, d::Real, x::Real)

Return the cdf of the marginal distribution for duration `d` of the model `pd` evaluated at `x`.
"""
function cdf(pd::MarginalScalingModel, d::Real, x::Real)

    margdist = IDFCurves.getdistribution(pd, d)
    return Distributions.cdf(margdist, x)

end

"""
    cdf(pd::MarginalScalingModel, d::Real, x::AbstractVector{<:Real})

Return the vector of the cdf of the marginal distribution for duration `d` of the model `pd` evaluated at every point in vector `x`.
"""
function cdf(pd::MarginalScalingModel, d::Real, x::AbstractVector{<:Real})

    margdist = IDFCurves.getdistribution(pd, d)
    return cdf.(margdist, x)

end

"""
    loglikelihood(pd::MarginalScalingModel, data::IDFdata)

 Return the loglikelihood of the parameters in `pd` as a function of `data``    
"""
function loglikelihood(pd::MarginalScalingModel, data::IDFdata)
    
    ll = 0.

    for tag in gettag(data)

        marginal = getdistribution(pd, getduration(data, tag))

        ll += sum(logpdf.(marginal, getdata(data, tag)))

    end
    
    return ll
    
end

"""
    quantile(pd::MarginalScalingModel, d::Real, p::Real)

Compute the quantile of level `p` for the duration `d` of the scaling model `pd`. 
"""
function quantile(pd::MarginalScalingModel, d::Real, p::Real)
    @assert 0<p<1 "The quantile level p must be in (0,1)."
    @assert d>0 "The duration must be positive."

    marginal = IDFCurves.getdistribution(pd, d)

    return Distributions.quantile(marginal, p)

end


"""
    rand(pd::MarginalScalingModel, d::AbstractVector{<:Real}, n::Int=1, ; tags::AbstractVector{<:AbstractString}=String[], x::AbstractVector{<:Real}=Float64[])

Generate a random sample of size `n` for duration vector `d` from the scaling model `pd`.
    
### Details

Duration tags and time vector can be provided with the keyword argument `tags` and `x` respectively. 
"""
function rand(pd::MarginalScalingModel, d::AbstractVector{<:Real}, n::Int=1, ; tags::AbstractVector{<:AbstractString}=String[], x::AbstractVector{<:Real}=Float64[])
    
    m = length(d)
    
    if isempty(tags)
        tags = string.(1:m)
    else
        @assert length(tags) == length(d) "Duration tag length must match the duration vector length."
    end

    if isempty(x)
        x = collect(1:n)
    else
        @assert length(tags) == length(d) "X tick length must match the sample size n."
    end

    x_dict = Dict(zip(tags, repeat([x], m)))
    d_dict = Dict(zip(tags, d))

    marginals = getdistribution.(pd, d)
    
    y = rand.(marginals, n)

    data_dict = Dict(zip(tags, y))
    
    return IDFdata(tags, d_dict, x_dict, data_dict)
    
end

"""
    scalingtype(fd::MarginalScalingModel)

Return the MarginalScalingModel specific type
"""
function scalingtype(fd::MarginalScalingModel)
    
    return eval(nameof(typeof(fd)))
    
end

### Fit


function fit_mle(pd_type::Type{<:MarginalScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractVector{<:Real})

    fitted_global_model = fit_mle(DependentScalingModel{pd_type, UncorrelatedStructure, IdentityCopula}, data, d₀, initialvalues)
    
    return getmarginalmodel(fitted_global_model)

end

function fit_mle(pd_type::Type{<:MarginalScalingModel}, data::IDFdata, d₀::Real)

    fitted_global_model = fit_mle(DependentScalingModel{pd_type, UncorrelatedStructure, IdentityCopula}, data, d₀)
    
    return getmarginalmodel(fitted_global_model)

end

"""

    hessian(fd::MarginalScalingModel, data::IDFdata)

Compute the Hessian matrix of the loglikelihood of the fitted scaling model `fd` associated with the IDF data `data`.
"""
function hessian(fd::MarginalScalingModel, data::IDFdata)

    return hessian(DependentScalingModel(fd, UncorrelatedStructure(), IdentityCopula), data)

end

"""
    godambe(fd::GeneralScaling, data::IDFdata)

Compute the Godambe matrix from the GeneralScaling model `fd` and the IDFData `data`.

## Detail

See also [`godambe`](@ref).
"""
function godambe(fd::MarginalScalingModel, data::IDFdata)
    
    J = variability_matrix(fd, data)
    
    H = IDFCurves.hessian(fd, data)
    
    G = PDMat(PDMats.X_invA_Xt(J, H))
    
    return G
    
end

"""
    quantilevar(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real)

Compute with the Delta method the quantile of level `p` variance for the duration `d` of the fitted scaling model `fd` on the IDFdata `data`.      
"""
function quantilevar(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real)
    
    return quantilevar(DependentScalingModel(fd, UncorrelatedStructure(), IdentityCopula), data, d, p)

end

"""
    quantilevar(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, H::PDMat{<:Real})

Compute with the Delta method the quantile of level `p` variance for the duration `d` of the fitted scaling model `fd` on the IDFdata `data`.

## Details

This function uses the Hessian matrix `H` provided in the argument. 
"""
function quantilevar(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, H::PDMat{<:Real})
    
    return quantilevar(DependentScalingModel(fd, UncorrelatedStructure(), IdentityCopula), data, d, p, H)

end

"""
    quantilecint(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)

Compute the approximate Wald quantile confidence interval of level (1-`α`) of the quantile of level `q` for the duration `d`.
"""
function quantilecint(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)
    
    return quantilecint(DependentScalingModel(fd, UncorrelatedStructure(), IdentityCopula), data, d, p, α)

end


"""
    variability_matrix(fd::MarginalScalingModel, data::IDFdata)

Estimate the variability matrix for the MarginalScalingModel `fd` according to the IDFData `data`.
    
## Detail

The data matrix is constructied assuming that all durations have the same number of observations.

See also [`variability_matrix`](@ref).
"""
function variability_matrix(fd::MarginalScalingModel, data::IDFdata)
   
    # TODO: Handling missing data. Assuming here that all durations have the same number of observations.
    Y = stack(getdata.(data, gettag(data)), dims=1)
    
    θ̂ = collect(params(fd))
        
    d, n = size(Y)
    
    J = zeros(length(θ̂), length(θ̂))
        
    scaling_model = scalingtype(fd)
        
    pd(θ) = getdistribution.(scaling_model(IDFCurves.duration(fd), θ...), getduration.(data, gettag(data)))
    ll(θ, y) = sum(logpdf.(pd(θ), y))
    u(θ, y) = ForwardDiff.gradient( θ -> ll(θ, y), θ)
    
    for y in eachcol(Y)
        
        uᵢ = u(θ̂, y)
        
        J .+= uᵢ * uᵢ'
    end
    
    return PDMat(Symmetric(J))
        
end
    