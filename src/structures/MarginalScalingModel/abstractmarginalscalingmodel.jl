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

    hessian(fd::MarginalScalingModel, data::IDFdata)

Compute the Hessian matrix of the loglikelihood of the fitted scaling model `pd` associated with the IDF data `data`.
"""
function hessian(fd::MarginalScalingModel, data::IDFdata)

    d₀ = duration(fd)
    θ̂ = collect(params(fd))

    T = eval(nameof(typeof(fd)))

    fobj(θ) = -loglikelihood(T(d₀, θ...), data)

    H = Hermitian(ForwardDiff.hessian(fobj, θ̂))

end

"""
    quantilevar(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real)

Compute with the Delta method the quantile of level `p` variance for the duration `d` of the fitted scaling model `fd` on the IDFdata `data`.      
"""
function quantilevar(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration should be positive."

    d₀ = duration(fd)
    θ̂ = collect(params(fd))

    H = IDFCurves.hessian(fd, data)

    T = eval(nameof(typeof(fd)))

    # quantile function
    g(θ::DenseVector{<:Real}) = quantile(T(d₀, θ...), d, p)

    # gradient
    ∇ = ForwardDiff.gradient(g, θ̂)

    # Approximate variance computed with the delta method
    u = H\∇
    v = dot(∇, u)

    return v

end

"""
    quantilecint(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)

Compute the approximate Wald quantile confidence interval of level (1-`α`) of the quantile of level `q` for the duration `d`.
"""
function quantilecint(fd::MarginalScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration sould be positive."
    @assert 0<α<1 "the confidence level (1-α) should be in (0,1)."

    q̂ = quantile(fd, d, p)
    v = IDFCurves.quantilevar(fd, data, d, p)

   dist = Normal(q̂, sqrt(v))

   return quantile.(dist, [α/2, 1-α/2])

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


### Fit


function fit_mle(pd_type::Type{<:MarginalScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractVector{<:Real})

    fitted_global_model = fit_mle(DependentScalingModel{pd_type, UncorrelatedStructure, IdentityCopula}, data, d₀, initialvalues)
    
    return getmarginalmodel(fitted_global_model)

end