"""
    dGEV(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real)

Construct a dGEV distribution type.

## Details

**TODO**

``\\mu_d = \\mu_0 \\left(\\frac{d+\\delta}{d_0+\\delta} \\right) ^{-\\alpha} \\qquad 
  \\sigma_d = \\sigma_0 \\left(\\frac{d+\\delta}{d_0+\\delta} \\right) ^{-\\alpha} \\qquad 
  \\xi_d = \\xi``

## References

Koutsoyiannis, D., Kozonis, D. and Manetas, A. (1998). 
A mathematical framework for studying rainfall intensity-duration-frequency relationships,
*Journal of Hydrology*, 206(1-2), 118-135, https://doi.org/10.1016/S0022-1694(98)00097-3.
"""
struct dGEV <: AbstractScalingModel
    d₀::Real # reference duration
    μ₀::Real 
    σ₀::Real
    ξ::Real
    α::Real # duration exponent (defining slope of the IDF curve)
    δ::Real # duration offset (defining curvature of the IDF curve)

    function dGEV(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real)
        
        @assert δ ≥ 0 "Duration offset must be non-negative"
        @assert 0 < α ≤ 1 "Duration exponent must be between 0 and 1"
        @assert σ₀ > 0 "Scale must be positive"
        
        return new(d₀, μ₀, σ₀, ξ, α, δ)
        
    end
end

Base.Broadcast.broadcastable(obj::dGEV) = Ref(obj)


### Parameters

"""
    duration(pd::dGEV)

Return the reference duration.
"""
duration(pd::dGEV) = pd.d₀

"""
    exponent(pd::dGEV)

Return the duration exponent.
"""
exponent(pd::dGEV) = pd.α

location(pd::dGEV) = pd.μ₀

"""
    offset(pd::dGEV)

Return the duration offset
"""
offset(pd::dGEV) = pd.δ

params(pd::dGEV) = (location(pd), scale(pd), shape(pd), exponent(pd), offset(pd))

scale(pd::dGEV) = pd.σ₀

shape(pd::dGEV) = pd.ξ



### Methods

"""
    getdistribution(pd::dGEV, d::Real)

Return the marginal GEV distribution for duration `d`.
"""
function getdistribution(pd::dGEV, d::Real)
    
    μ₀ = location(pd)
    σ₀ = scale(pd)
    ξ = shape(pd)
    α = exponent(pd)
    δ = offset(pd)
    
    d₀ = duration(pd)
    
    ls = -α * (log(d + δ) - log(d₀ + δ))
    s = exp(ls)

    μ = μ₀ * s
    σ = σ₀ * s
    
    return GeneralizedExtremeValue(μ, σ, ξ)
    
end

"""

    hessian(pd::dGEV, data::IDFdata)

Compute the Hessian matrix of the pGEV distribution `pd` associated with the IDF data `data`.
"""
function hessian(pd::dGEV, data::IDFdata)

    d₀ = duration(pd)
    θ̂ = [params(pd)...]

    fobj(θ) = -loglikelihood(dGEV(d₀, θ...), data)

    H = Hermitian(ForwardDiff.hessian(fobj, θ̂))

end

"""
    loglikelihood(pd::dGEV, data::IDFdata)

 Return the loglikelihood of the parameters in `pd` as a function of `data``    
"""
function loglikelihood(pd::dGEV, data::IDFdata)
    
    ll = 0.

    for tag in gettag(data)

        marginal = getdistribution(pd, getduration(data, tag))

        ll += sum(logpdf.(marginal, getdata(data, tag)))

    end
    
    return ll
    
end

"""
    map_to_real_space(::Type{<:dGEV}, θ)

Map the vector of parameters of the dGEV to place them in the real hypercube.
"""
function map_to_real_space(::Type{<:dGEV}, θ)
    @assert length(θ) == 5 "The parameter vector length must be 5. Verify that the reference duration is not included."

    return [θ[1], exp(θ[2]), logistic(θ[3] + .5), logistic(θ[4]), exp(θ[5])]

end

"""
    quantile(pd::dGEV, d::Real, p::Real)

Compute the quantile of level `p` for the duration `d` of the dGEV model `pd`. 
"""
function quantile(pd::dGEV, d::Real, p::Real)
    @assert 0<p<1 "The quantile level p must be in (0,1)."
    @assert d>0 "The duration must be positive."

    marginal = IDFCurves.getdistribution(pd, d)

    return quantile(marginal, p)

end

"""
    quantilevar(fd::dGEV, data::IDFdata, d::Real, p::Real)

Compute with the Delta method the quantile of level `p` variance for the duration `d` of the fitted dGEV model `fd` on the IDFdata `data`.      
"""
function quantilevar(fd::dGEV, data::IDFdata, d::Real, p::Real)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration sould be positive."

    d₀ = duration(fd)
    θ̂ = collect(params(fd))

    H = IDFCurves.hessian(fd, data)

    # quantile function
    g(θ::DenseVector{<:Real}) = quantile( dGEV(d₀, θ...), d, p)

    # gradient
    ∇ = ForwardDiff.gradient(g, θ̂)

    # Approximate variance computed with the delta method
    u = H\∇
    v = dot(∇, u)

    return v

end

"""
    quantilecint(fd::dGEV, data::IDFdata, d::Real, p::Real, α::Real=.05)

Compute the approximate Wald quantile confidence interval of level (1-`α`) of the quantile of level `q` for the duration `d`.
"""
function quantilecint(fd::dGEV, data::IDFdata, d::Real, p::Real, α::Real=.05)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration sould be positive."
    @assert 0<α<1 "the confidence level (1-α) should be in (0,1)."

    q̂ = quantile(fd, d, p)
    v = IDFCurves.quantilevar(fd, data, d, p)

   dist = Normal(q̂, sqrt(v))

   return quantile.(dist, [α/2, 1-α/2])

end



"""
    rand(pd::dGEV, d::AbstractVector{<:Real}, n::Int=1, ; tags::AbstractVector{<:AbstractString}=String[], x::AbstractVector{<:Real}=Float64[])

Generate a random sample of size `n` for duration vector `d` from the dGEV model `pd`.
    
### Details

Duration tags and time vector can be provided with the keyword argument `tags` and `x` respectively. 
"""
function rand(pd::dGEV, d::AbstractVector{<:Real}, n::Int=1, ; tags::AbstractVector{<:AbstractString}=String[], x::AbstractVector{<:Real}=Float64[])
    
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
    Base.show(io::IO, obj::dGEV)

Override of the show function for the objects of type dGEV.

"""
function Base.show(io::IO, obj::dGEV)
    println(io, "dGEV(d₀ = ", duration(obj),
        ", μ₀ = ", round(location(obj), digits=4),
        ", σ₀ = ", round(scale(obj), digits=4),
        ", ξ = ", round(shape(obj), digits=4),
        ", α = ", round(exponent(obj), digits=4),
        ", δ = ", round(offset(obj), digits=4),
        ")")
end




## Fit

function fit_mle_gradient_free(pd::Type{<:dGEV}, data::IDFdata, d₀::Real, initialvalues::AbstractVector{<:Real})

    fobj(θ::DenseVector{<:Real}) = -loglikelihood(dGEV(d₀, map_to_real_space(dGEV, θ)...), data)
    
    res = Optim.optimize(fobj, initialvalues)
    
    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end
    
    fd = dGEV(d₀, map_to_real_space(dGEV, θ̂)...)
    
    return fd
end


function fit_mle(pd::Type{<:dGEV}, data::IDFdata, d₀::Real, initialvalues::AbstractVector{<:Real})

    fobj(θ::DenseVector{<:Real}) =  -loglikelihood(dGEV(d₀, map_to_real_space(dGEV, θ)...), data)

    res = Optim.optimize(TwiceDifferentiable(fobj, initialvalues; autodiff = :forward), initialvalues)
    
    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = initialvalues
    end
    
    fd = dGEV(d₀, map_to_real_space(dGEV, θ̂)...)
    
    return fd
end
