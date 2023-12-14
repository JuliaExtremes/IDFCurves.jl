"""
    qqplot(pd::Distribution, y::Vector{Float64})

Quantile-Quantile plot from ExtendedExtremes.jl
"""
function qqplot(pd::Distribution, y::Vector{Float64})

    n = length(y)
    q = sort(y)

    p = (1:n) ./ (n+1)

    q̂ = quantile.(pd, p);
    
    l1 = layer(y = q, x = q̂, Geom.point)
    l2 = layer(y = q[[1, end]], x = q[[1, end]], Geom.line, Theme(default_color="red", line_style=[:dash]))

    return plot(l2, l1,
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Quantile Plot"),
        Theme(discrete_highlight_color=c->nothing))
end

"""
    qqplot(pd::Distribution, y::Vector{Float64})

Quantile-Quantile plot from ExtendedExtremes.jl
"""
function qqplot(pd::dGEV, data::IDFdata, d::Real)
    @assert d>0 "duration must be positive."

    marginal = getdistribution(pd, d)
    
    y = getdata(data, d)

    qqplot(marginal, y)
end

"""
    qqplotci(fm::dGEV, α::Real=.05)

Quantile-Quantile plot along with the confidence/credible interval of level `1-α`.
```
 
"""
function qqplotci(fd::dGEV, data::IDFdata, d::Real, α::Real=.05)
    @assert d>0 "duration must be positive."
    @assert 0 < α < 1 "the level should be in (0,1)." 

    tag = gettag(data, d)

    y = getdata(data, tag)

    n = length(y)
    q = sort(y)

    p = (1:n) ./ (n+1)

    q̂ = quantile.(fd, d, p)

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



function qqplotci(fd::DependentScalingModel, data::IDFdata, d::Real, α::Real=.05)
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
