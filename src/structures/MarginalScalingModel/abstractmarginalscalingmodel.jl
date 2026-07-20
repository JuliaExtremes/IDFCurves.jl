abstract type MarginalScalingModel <: ContinuousMultivariateDistribution end

include("simplescalingmodel.jl")
include("generalscalingmodel.jl")
include("universalscalingmodel.jl")


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
    getdistribution(sm::MarginalScalingModel, d::Real)

Return the marginal GEV distribution for duration `d` under the scaling model `sm`.
"""
function getdistribution(sm::MarginalScalingModel, d::Real)

    s = scaling_factor(sm, d)

    μ = location(sm) * s
    σ = scale(sm) * s
    ξ = shape(sm)

    return GeneralizedExtremeValue(μ, σ, ξ)
    
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
    rand(
        rng::AbstractRNG,
        model::MarginalScalingModel,
        duration_dict::AbstractDict{String,<:Real},
        n::Integer=1,
    )

    rand(
        model::MarginalScalingModel,
        duration_dict::AbstractDict{String,<:Real},
        n::Integer=1,
    )

Generate an `IDFdata` sample of size `n` from `model` at the durations specified
by `duration_dict`.

Samples are generated independently at each duration. The first method uses the
provided random-number generator, while the second uses `Random.default_rng()`.
"""
function rand end

function rand(rng::Random.AbstractRNG, model::MarginalScalingModel, duration_dict::AbstractDict{String,<:Real}, n::Integer=1)

    n > 0 || throw(ArgumentError("n must be positive."))

    tags = sort!(
        collect(keys(duration_dict));
        by=tag -> duration_dict[tag],
    )

    durations = Dict{String,Float64}(
        tag => Float64(duration_dict[tag]) for tag in tags
    )

    years = Dict{String,Vector{Int}}(
        tag => collect(1:n) for tag in tags
    )

    observations = Dict{String,Vector{Float64}}()

    for tag in tags
        d = durations[tag]

        d > 0 || throw(ArgumentError(
            "Duration corresponding to tag=$tag must be positive, got $d.",
        ))

        pd = getdistribution(model, d)
        observations[tag] = rand(rng, pd, n)
    end

    return IDFdata(tags, durations, years, observations)
end

function rand(model::MarginalScalingModel, duration_dict::AbstractDict{String,<:Real}, n::Integer=1)

    rng=Random.default_rng()

    return rand(rng, model, duration_dict, n)
end

"""
    scalingtype(fd::MarginalScalingModel)

Return the MarginalScalingModel specific type
"""
function scalingtype(fd::MarginalScalingModel)
    
    return eval(nameof(typeof(fd)))
    
end

### Fit

"""
    fit_mle(::Type{<:MarginalScalingModel}, data::IDFdata, initialmodel::MarginalScalingModel)

Fit a marginal scaling model by maximum likelihood using `initialmodel` for initialization.
"""
function fit_mle(pd::Type{<:MarginalScalingModel}, data::IDFdata, initialmodel::MarginalScalingModel)

    IDFCurves.scalingtype(initialmodel) === pd || 
        throw(ArgumentError("Model and initial model must be of the same type, got $pd ≠ $(IDFCurves.scalingtype(initialmodel))"))

    d₀ = duration(initialmodel)

    initialvalues = collect(params(initialmodel))

    θ₀ = map_to_real_space(pd, initialvalues)

    model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(pd, d₀, θ)
    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    isfinite(fobj(θ₀)) || 
        throw(ArgumentError("The initial model has a nonfinite log-likelihood. At least one observation may lie outside its support."))
    
    res = Optim.optimize(fobj, θ₀)

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = θ₀
    end

    return model(θ̂)

end

"""
    fit_mle(::Type{<:MarginalScalingModel}, data::IDFdata, d₀::Real)

Fit a marginal scaling model by maximum likelihood using automatically generated initial values.
"""
function fit_mle(pd::Type{<:MarginalScalingModel}, data::IDFdata, d₀::Real)

    (d₀ > 0) ||  throw(ArgumentError("Reference duration must be positive, got d₀=$d₀"))

    initialmodel = initialize(pd, data, d₀)

    return fit_mle(pd, data, initialmodel)

end



"""

    hessian(fd::MarginalScalingModel, data::IDFdata)

Compute the Hessian matrix of the loglikelihood of the fitted scaling model `fd` associated with the IDF data `data`.
"""
function hessian(fm::MarginalScalingModel, data::IDFdata)

    T = scalingtype(fm)
    d₀ = duration(fm)
    θ̂ = collect(params(fm))

    model(θ::DenseVector{<:Real}) = T(d₀, θ...)
    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    H = ForwardDiff.hessian(fobj, θ̂)

    return PDMat(Symmetric(H))

end

#TODO Specialize the methods for MarginalScalingModel without using DependentScalingModel

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
    quantilecint(pd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, H::PDMat{<:Real}, α::Real=.05)

Compute the approximate Wald quantile confidence interval of level (1-`α`) of the quantile of level `q` for the duration `d`.

## Details

This function uses the Hessian matrix `H` provided in the argument.  
"""
function quantilecint(pd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, H::PDMat{<:Real}, α::Real=.05)
    
    return quantilecint(DependentScalingModel(pd, UncorrelatedStructure(), IdentityCopula), data, d, p, H, α)

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
    