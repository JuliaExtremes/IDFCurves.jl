
struct DependentScalingModel{T₁<:MarginalScalingModel, T₂<:CorrelationStructure, T₃}
    marginal::T₁
    correlogram::T₂
    DependentScalingModel(m::MarginalScalingModel, c::CorrelationStructure, T₃::Type{<:EllipticalCopula}) = new{typeof(m), typeof(c), T₃}(m,c)
end


Base.Broadcast.broadcastable(obj::DependentScalingModel) = Ref(obj)

function getcopulatype(pd::Type{DependentScalingModel{T₁, T₂, T₃}}) where {T₁, T₂, T₃}
    return T₃
end

function getcopulatype(pd::DependentScalingModel{T₁, T₂, T₃}) where {T₁, T₂, T₃}
    return T₃
end

function getmarginaltype(pd::Type{DependentScalingModel{T₁, T₂, T₃}}) where {T₁, T₂, T₃}
    return eval(nameof(T₁))
end

function getmarginalmodel(pd::DependentScalingModel)
    return pd.marginal
end

function getcorrelogramtype(pd::Type{DependentScalingModel{T₁, T₂, T₃}}) where {T₁, T₂, T₃}
    return eval(nameof(T₂))
end

function getcorrelogram(pd::DependentScalingModel)
    return pd.correlogram
end

"""
    duration(pd::DependentScalingModel)

Return the reference duration.
"""
duration(pd::DependentScalingModel) = duration(getmarginalmodel(pd))

function params(pd::DependentScalingModel) 

    scaling_model = getmarginalmodel(pd)
    cor_model = getcorrelogram(pd)
    return (params(scaling_model)..., params(cor_model)...)

end


function getabstracttype(pd::DependentScalingModel)
    T = typeof(pd)
    return DependentScalingModel{getmarginaltype(T), getcorrelogramtype(T), getcopulatype(T)}
end


"""
    quantile(pd::DependentScalingModel, d::Real, p::Real)

Compute the quantile of level `p` for the duration `d` of the scaling model `pd`. 
"""
function quantile(pd::DependentScalingModel, d::Real, p::Real)
    @assert 0<p<1 "The quantile level p must be in (0,1)."
    @assert d>0 "The duration must be positive."

    return quantile( getmarginalmodel(pd), d, p)

end

function loglikelihood(pd::DependentScalingModel, data)

    tags = gettag(data)
    idx = getyear(data, tags[1])
    d = getduration.(data, tags)
    h = IDFCurves.logdist(d) 

    y = getdata.(data, tags, idx')

    # Marginal loglikelihood
    ll = loglikelihood(getmarginalmodel(pd), data)

    Σ = cor.(getcorrelogram(pd), h)
    C = IDFCurves.getcopulatype(pd)(Σ)

    u = cdf.(getmarginalmodel(pd), d, y)
    for c in eachcol(u)
        ll += IDFCurves.logpdf(C, c)
    end

    return ll

end

"""
    construct_model(::Type{<:DependentScalingModel}, d₀, θ)

Construct a DependentScalingModel from a set of transformed parameters θ in the real space.
"""
function construct_model(pd::Type{<:DependentScalingModel}, d₀::Real, θ::DenseVector{<:Real})

    scaling_model = IDFCurves.getmarginaltype(pd)
    copula_model = IDFCurves.getcopulatype(pd)
    correlogram_model = IDFCurves.getcorrelogramtype(pd)

    k_marginal, k_correlation = params_number(scaling_model), params_number(correlogram_model)
    @assert length(θ) == k_marginal + k_correlation "Length of θ ("*string(length(θ))*") is wrong. Should match the total number of parameters for the model ("*string(k_marginal + k_correlation)*")."

    sm = IDFCurves.construct_model(scaling_model, d₀, θ[1:k_marginal])
    Σ = IDFCurves.construct_model(correlogram_model, θ[(k_marginal+1):(k_marginal+k_correlation)])

    return DependentScalingModel(sm, Σ, copula_model)
    
end

"""
    map_to_real_space(::Type{<:DependentScalingModel}, θ)

Map the parameters from the DependentScalingModel parameter space to the real space.
"""
function map_to_real_space(pd::Type{<:DependentScalingModel}, θ::AbstractVector{<:Real})
    scaling_model = IDFCurves.getmarginaltype(pd)
    correlogram_model = IDFCurves.getcorrelogramtype(pd)
    k_marginal, k_correlation = params_number(scaling_model), params_number(correlogram_model)
    @assert length(θ) == k_marginal + k_correlation "Length of the parameter vector ("*string(length(θ))*") is wrong. Should match the total number of parameters for the model ("*string(k_marginal + k_correlation)*")."

    return [IDFCurves.map_to_real_space(scaling_model, θ[1:k_marginal])..., 
                IDFCurves.map_to_real_space(correlogram_model, θ[(k_marginal+1):(k_marginal+k_correlation)])...]

end


function fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractArray{<:Real})

    if abs(initialvalues[3]) < 0.0001 # the shape parameter can't be initalized at 0.0
        initialvalues[3] = 0.0001
    end

    θ₀ = map_to_real_space(pd, initialvalues)

    model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(pd, d₀, θ)
    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)
    @assert fobj(θ₀) < Inf "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    # function grad_fobj(G, θ)
    #     grad = ForwardDiff.gradient(fobj, θ)
    #     for i in eachindex(G)
    #         G[i] = grad[i]
    #     end
    # end
    # function hessian_fobj(H, θ)
    #     hess = ForwardDiff.hessian(fobj, θ)
    #     for i in axes(H,1)
    #         for j in axes(H,2)
    #             H[i,j] = hess[i,j]
    #         end
    #     end
    # end

    grad_fobj, hessian_fobj = compute_derivatives(fobj)

    # optimization
    res = nothing
    try 
        res = Optim.optimize(fobj, grad_fobj, hessian_fobj, θ₀)
        @assert Optim.converged(res)
    catch e
        res = Optim.optimize(fobj, θ₀)
    end

    if Optim.converged(res)
        θ̂ = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        θ̂ = θ₀
    end

    return model(θ̂)

end

"""
    initialize(::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real)

Initialize a vector of parameters for the DependentScalingmodel adapted to the data.
The initialization is done independently for the marginal scaling model and the correlation structure.
"""
function initialize(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real)

    scaling_model = IDFCurves.getmarginaltype(pd)
    correlogram_model = IDFCurves.getcorrelogramtype(pd)

    init_scaling_params = initialize(scaling_model, data, d₀)
    init_corr_params = initialize(correlogram_model, data)

    return [init_scaling_params ; init_corr_params]
end

"""
    fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real)

Fits a DependentScalingModel of type pd to the data using automatic initialization.
"""
function fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real)

    initialvalues = initialize(pd, data, d₀)

    return fit_mle(pd, data, d₀, initialvalues)

end

"""

    hessian(pd::DependentScalingModel, data::IDFdata)

Compute the Hessian matrix of the DependentScalingModel distribution `pd` associated with the IDF data `data`.
"""
function hessian(pd::DependentScalingModel, data::IDFdata)

    T = getabstracttype(pd)
    d₀ = duration(pd)
    θ̂ = collect(params(pd))
    
    fobj(θ::DenseVector{<:Real}) = -loglikelihood(IDFCurves.construct_model(T, d₀, map_to_real_space(T, θ)), data)

    H = ForwardDiff.hessian(fobj, θ̂)

    return PDMat(Symmetric(H))

end


"""
    quantilevar(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real)

Compute with the Delta method the quantile of level `p` variance for the duration `d` of the fitted model `pd` on the IDFdata `data`.      
"""
function quantilevar(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration should be positive."

    T = getabstracttype(pd)

    θ̂ = collect(params(pd))
    d₀ = duration(getmarginalmodel(pd))

    # quantile function
    function g(θ::DenseVector{<:Real}) 
        model = IDFCurves.construct_model(T, d₀, map_to_real_space(T, θ))
        return quantile(model, d, p)
    end

    H = hessian(pd, data)

    v = Extremes.delta(g, θ̂, H)

    return v

end



"""
    quantilecint(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)

Compute the approximate Wald quantile confidence interval of level (1-`α`) of the quantile of level `p` for the duration `d`.
"""
function quantilecint(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration sould be positive."
    @assert 0<α<1 "the confidence level (1-α) should be in (0,1)."

    q̂ = quantile(pd, d, p )
    v = quantilevar(pd, data, d, p)

    if v > 0
        dist = Normal(q̂, sqrt(v))
        return quantile.(dist, [α/2, 1-α/2])
    else
        return [q̂, q̂]
    end

end