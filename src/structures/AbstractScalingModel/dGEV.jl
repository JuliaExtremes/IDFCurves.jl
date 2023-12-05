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

    H = ForwardDiff.hessian(fobj, θ̂)

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
