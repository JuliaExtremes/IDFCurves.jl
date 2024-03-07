
struct DependentScalingModel{T₁<:MarginalScalingModel, T₂<:CorrelationStructure, T₃}
    marginal::T₁
    correlogram::T₂
    DependentScalingModel(m, c, T₃::Type{<:EllipticalCopula}) = new{typeof(m), typeof(c), T₃}(m,c)
end

Base.Broadcast.broadcastable(obj::DependentScalingModel) = Ref(obj)



function getcopulatype(obj::Type{DependentScalingModel{T₁, T₂, T₃}}) where {T₁, T₂, T₃}
    return T₃
end

function getcopulatype(pd::DependentScalingModel{T₁, T₂, T₃}) where {T₁, T₂, T₃}
    return T₃
end

# function getcopula(pd::DependentScalingModel)
#     return pd.copula
# end

function getmarginaltype(obj::Type{DependentScalingModel{T₁, T₂, T₃}}) where {T₁, T₂, T₃}
    return T₁
end

function getmarginalmodel(pd::DependentScalingModel)
    return pd.marginal
end

function getcorrelogramtype(obj::Type{DependentScalingModel{T₁, T₂, T₃}}) where {T₁, T₂, T₃}
    return eval(nameof(T₂))
end

function getcorrelogram(pd::DependentScalingModel)
    return pd.correlogram
end

function construct_model(obj::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, θ::DenseVector{<:Real})
    # TODO Verify number of parameter with the model

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations) 

    scaling_model = IDFCurves.getmarginaltype(obj)
    copula_model = IDFCurves.getcopulatype(obj)
    correlogram_model = IDFCurves.getcorrelogramtype(obj)

    sm = scaling_model(d₀, IDFCurves.map_to_param_space(scaling_model, θ[1:5])...)
    Σ = correlogram_model(exp(θ[6]), exp(θ[7]))   #TODO make it general for other corelogram

    return DependentScalingModel(sm, Σ, copula_model)
    
end



function loglikelihood(pd::DependentScalingModel, data)

    tags = gettag(data)
    idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
    d = getduration.(data, tags)
    h = IDFCurves.logdist(d) 

    y = getdata.(data, tags, idx')

    # Marginal loglikelihood
    ll = loglikelihood(getmarginalmodel(pd), data)

    Σ = cor.(getcorrelogram(pd), h)
    C = IDFCurves.getcopulatype(pd)(Σ) #TODO Make it general for TCopula

    u = cdf.(getmarginalmodel(pd), d, y)
    for c in eachcol(u)
        ll += IDFCurves.logpdf(C, c)
    end

    return ll

end


function fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractArray{<:Real})

    if initialvalues[3] == 0.0 # the shape parameter can't be initalized at 0.0
        initialvalues[3] = 0.0001
    end

    θ₀ = [IDFCurves.map_to_real_space(IDFCurves.getmarginaltype(pd), initialvalues[1:5])..., log(initialvalues[6]), log(initialvalues[7])]

    model(θ::DenseVector{<:Real}) = IDFCurves.construct_model(pd, data, d₀, θ)

    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    res = Optim.optimize(fobj, θ₀)

    θ̂ = Optim.minimizer(res)

    return model(θ̂)

end


"""

    hessian(pd::DependentScalingModel, data::IDFdata)

Compute the Hessian matrix of the DependentScalingModel distribution `pd` associated with the IDF data `data`.
"""
function hessian(pd::DependentScalingModel, data::IDFdata)

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations)

    θ̂ = [params(getmarginalmodel(pd))..., params(getcorrelogram(pd))...]

    d₀ = duration(getmarginalmodel(pd))

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