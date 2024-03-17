
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
    return T₁
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


function construct_model(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, θ::DenseVector{<:Real})

    scaling_model = IDFCurves.getmarginaltype(pd)
    copula_model = IDFCurves.getcopulatype(pd)
    correlogram_model = IDFCurves.getcorrelogramtype(pd)

    k_marginal, k_correlation = params_number(scaling_model), params_number(correlogram_model)
    @assert length(θ) == k_marginal + k_correlation "Length of θ ("*string(length(θ))*") is wrong. Should match the total number of parameters for the model ("*string(k_marginal + k_correlation)*")."

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations) 

    sm = scaling_model(d₀, IDFCurves.map_to_param_space(scaling_model, θ[1:k_marginal])...)
    Σ = correlogram_model(IDFCurves.map_to_param_space(correlogram_model, θ[(k_marginal+1):(k_marginal+k_correlation)])...)

    return DependentScalingModel(sm, Σ, copula_model)
    
end


function fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractArray{<:Real})

    if initialvalues[3] == 0.0 # the shape parameter can't be initalized at 0.0
        initialvalues[3] = 0.0001
    end

    scaling_model = IDFCurves.getmarginaltype(pd)
    correlogram_model = IDFCurves.getcorrelogramtype(pd)
    k_marginal, k_correlation = params_number(scaling_model), params_number(correlogram_model)
    @assert length(initialvalues) == k_marginal + k_correlation "Length of the initial vector of parameters ("*string(length(initialvalues))*") is wrong. Should match the total number of parameters for the model ("*string(k_marginal + k_correlation)*")."
    θ₀ = [IDFCurves.map_to_real_space(scaling_model, initialvalues[1:k_marginal])..., 
            IDFCurves.map_to_real_space(correlogram_model, initialvalues[(k_marginal+1):(k_marginal+k_correlation)])...] 

    model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(pd, data, d₀, θ)
    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)
    @assert fobj(θ₀) < Inf "The initial value vector is not a member of the set of possible solutions. At least one data lies outside the distribution support."

    function grad_fobj(G, θ)
        grad = ForwardDiff.gradient(fobj, θ)
        for i in eachindex(G)
            G[i] = grad[i]
        end
    end
    function hessian_fobj(H, θ)
        hess = ForwardDiff.hessian(fobj, θ)
        for i in axes(H,1)
            for j in axes(H,2)
                H[i,j] = hess[i,j]
            end
        end
    end

    # optimization
    res = nothing
    try 
        res = Optim.optimize(fobj, grad_fobj, hessian_fobj, θ₀)
    catch e
        println("Gradient-descent algorithm could not converge - trying gradient-free optimization")
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

    hessian(pd::DependentScalingModel, data::IDFdata)

Compute the Hessian matrix of the DependentScalingModel distribution `pd` associated with the IDF data `data`.
"""
function hessian(pd::DependentScalingModel, data::IDFdata)

    scaling_model = IDFCurves.getmarginalmodel(pd)
    correlogram_model = IDFCurves.getcorrelogram(pd)

    θ̂ = [IDFCurves.map_to_real_space(typeof(scaling_model), [params(scaling_model)...])..., 
            IDFCurves.map_to_real_space(typeof(correlogram_model), [params(correlogram_model)...])...] #TODO for now bug
    d₀ = duration(scaling_model)

    model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(typeof(pd), data, d₀, θ)

    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    H = ForwardDiff.hessian(fobj, θ̂)

    return H

end


"""
    quantilevar(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real)

Compute with the Delta method the quantile of level `p` variance for the duration `d` of the fitted model `pd` on the IDFdata `data`.      
"""
function quantilevar(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration sould be positive."

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations)

    c = IDFCurves.getcormatrix(getcopula(pd))
    ρ̂ = -h[1,2]/log(c[1,2])

    scaling_model = typeof(IDFCurves.getmarginalmodel(pd))

    θ̂ = [collect(params(getmarginalmodel(pd))) ; ρ̂]
    d₀ = duration(getmarginalmodel(pd))

    H = hessian(pd, data)

    # quantile function
    g(θ::DenseVector{<:Real}) = quantile( scaling_model(d₀, θ[1:5]...), d, p)

    # gradient
    # ∇ = [ForwardDiff.gradient(g, θ̂)..., 0.]
    ∇ = ForwardDiff.gradient(g, θ̂)

    # Approximate variance computed with the delta method
    u = H\∇
    v = dot(∇, u)

    return v

end



"""
    quantilecint(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)

Compute the approximate Wald quantile confidence interval of level (1-`α`) of the quantile of level `q` for the duration `d`.
"""
function quantilecint(pd::DependentScalingModel, data::IDFdata, d::Real, p::Real, α::Real=.05)
    @assert 0<p<1 "the quantile level sould be in (0,1)."
    @assert d>0 "the duration sould be positive."
    @assert 0<α<1 "the confidence level (1-α) should be in (0,1)."

    q̂ = quantile(IDFCurves.getmarginalmodel(pd), d, p)
    v = quantilevar(pd, data, d, p)

    if v > 0
        dist = Normal(q̂, sqrt(v))
        return quantile.(dist, [α/2, 1-α/2])
    else
        return [q̂, q̂]
    end

end