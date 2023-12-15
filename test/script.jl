using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [20, 5, .04, .76, .07])



initialvalues = [20, 5, .04, .76, .07, 3]

fm = IDFCurves.fit_mle(DependentScalingModel{dGEV, GaussianCopula}, data, 1, initialvalues)

IDFCurves.qqplotci(fm, data, 6)


















function fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractArray{<:Real})

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations) #TODO: Include the covariogram and the distance in the struct

    scaling_model = IDFCurves.getmarginaltype(pd)
    copula_model = IDFCurves.getcopulatype(pd)

    θ₀ = [IDFCurves.map_to_real_space(scaling_model, initialvalues[1:5])..., log(initialvalues[6])]

    model(θ::DenseVector{<:Real}) = DependentScalingModel(
        scaling_model(d₀::Real, IDFCurves.map_to_param_space(scaling_model, θ[1:5])...),
        copula_model(exp.(-h./exp(θ[6])))
    )

    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    res = Optim.optimize(fobj, θ₀)

    θ̂ = Optim.minimizer(res)

    return model(θ̂)

end

fm = IDFCurves.fit_mle(DependentScalingModel{dGEV, GaussianCopula}, data, 1, initialvalues)


"""

    hessian(pd::DependentScalingModel, data::IDFdata)

Compute the Hessian matrix of the DependentScalingModel distribution `pd` associated with the IDF data `data`.
"""
function hessian(pd::DependentScalingModel, data::IDFdata)

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations)

    c = IDFCurves.getcormatrix(getcopula(pd))
    ρ̂ = -h[1,2]/log(c[1,2])

    θ̂ = [params(getmarginalmodel(pd))..., ρ̂]

    d₀ = duration(getmarginalmodel(pd))

    scaling_model = typeof(getmarginalmodel(fm))
    copula_model = typeof(getcopula(fm))

    model(θ::DenseVector{<:Real}) = DependentScalingModel(
        scaling_model(d₀::Real, θ[1:5]...),
        copula_model(exp.(-h./θ[6]))
    )

    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    H = ForwardDiff.hessian(fobj, θ̂)

    return H

end


IDFCurves.hessian(fm, data)


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

IDFCurves.quantilevar(fm, data, 1, .95)


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

IDFCurves.quantilecint(fm, data, 1, .95)



function qqplotci(fd::DependenceScalingModel, data::IDFdata, d::Real, α::Real=.05)
    @assert d>0 "duration must be positive."
    @assert 0 < α < 1 "the level should be in (0,1)." 

    tag = gettag(data, d)

    y = getdata(data, tag)

    n = length(y)
    q = sort(y)

    p = (1:n) ./ (n+1)

    scaling_model = IDFCurves.getmarginalmodel(fd)

    q̂ = quantile.(scaling_model, d, p)

    df = DataFrame(Empirical = q, Model = q̂)

    q_inf = Float64[]
    q_sup = Float64[]

    for pᵢ in p
        c = quantilecint(fd, data, d, pᵢ)
        push!(q_inf, c[1])
        push!(q_sup, c[2])
    end

    df[:,:Inf] = q_inf
    df[:,:Sup] = q_sup

    l1 = layer(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="black", style=:dash), 
        Theme(default_color="black", discrete_highlight_color=c->nothing))
    l2 = layer(df, x=:Model, ymin=:Inf, ymax=:Sup, Geom.ribbon, Theme(lowlight_color=c->"lightgray"))
    
    return plot(l1,l2, Guide.xlabel("Model"), Guide.ylabel("Empirical"), Theme(background_color="white"))
end





