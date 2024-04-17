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
function qqplot(pd::MarginalScalingModel, data::IDFdata, d::Real)
    @assert d>0 "duration must be positive."

    marginal = getdistribution(pd, d)
    
    y = getdata(data, d)

    qqplot(marginal, y)
end

"""
    qqplotci(fm::MarginalScalingModel, α::Real=.05)

Quantile-Quantile plot along with the confidence/credible interval of level `1-α`.
```
 
"""
function qqplotci(fd::MarginalScalingModel, data::IDFdata, d::Real, α::Real=.05)
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


"""
Les fonctions suivantes servent à représenter les courbes IDF à partir d'un modèle de scaling.
"""

function get_durations_labels(durations::Vector{<:Real})
    """Returns a vector of strings that wll be the labels given to the durations on the x-axis of the plotted IDF curve"""

    durations = sort(durations)
    durations_lower_1h = durations[durations .< 1]
    durations_bigger_1h = durations[durations .>= 1]

    labels = Vector{String}()
    if length(durations_lower_1h) >= 1
        push!(labels, string(Int64(round(durations_lower_1h[1]*60))) * "min" )
        if length(durations_lower_1h) >= 2
            for d in durations_lower_1h[2:end-1]
                push!(labels, string(Int64(round(d*60))))
            end
            push!(labels, string(Int64(round(durations_lower_1h[end]*60))) * "min" )
        end
    end
    if length(durations_bigger_1h) >= 1
        push!(labels, string(Int64(round(durations_bigger_1h[1]))) * "h" )
        if length(durations_bigger_1h) >= 2
            for d in durations_bigger_1h[2:end-1]
                push!( labels, string(Int64(round(d))) )
            end
            push!(labels, string(Int64(round(durations_bigger_1h[end]))) * "h" )
        end 
    end

    return labels
    
end

function plotIDFCurves(model::DependentScalingModel; 
        T_values::Vector{<:Real}=[2,5,10,25,50,100],
        durations::Vector{<:Real}=[1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24],
        d_min::Union{Real, Nothing} = nothing,
        d_max::Union{Real, Nothing} = nothing,
        y_ticks::Union{Vector{<:Real}, Nothing} = nothing)

    if isnothing(d_min)
        d_min = minimum(durations)
    end
    if isnothing(d_max)
        d_max = maximum(durations)
    end
    durations = [ [d_min] ; durations[d_min .< durations .&& durations .< d_max] ; [d_max ]]

    d_step = d_min/10

    T_values = sort(T_values)
    durations = sort(durations)

    data_return_levels = crossjoin( DataFrame(T = T_values),  DataFrame(d = d_min:d_step:d_max) )
    transform!(data_return_levels, [:T, :d] => ((x,y) -> quantile.(Ref(model), y, 1 .- 1 ./ x)) => :return_level)

    layers = []
    for T in reverse(T_values)
        data = data_return_levels[data_return_levels[:,:T] .== T, :]
        push!(layers, layer(data, x = :d, y = :return_level, color = :T, Geom.line()))
    end

    labels = get_durations_labels(durations)
    f_label(x) = labels[durations .≈ exp(x)][1]
    palette = [Scale.color_continuous().f((2*i-1)/(2*length(T_values))) for i in eachindex(T_values)]

    if isnothing(y_ticks)
    y_ticks = range(log(minimum(data_return_levels[:,:return_level])), log(maximum(data_return_levels[:,:return_level])), 6)
    end

    p = plot(layers..., Scale.x_log(labels = f_label), Scale.y_log(labels = y -> "$(round(exp(y)))"),
    Scale.color_discrete_manual(palette...),
    Guide.xticks(ticks = log.(durations)),
    Guide.yticks(ticks = y_ticks),
    Guide.xlabel("Rainfall duration"),
    Guide.ylabel("Rainfall intensity (mm/h)"),
    Guide.colorkey(title="Return period T (years)"),
    Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
        key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
    )

    return p

end
