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
    qqplotci(fm::DependentScalingModel, α::Real=.05, H::PDMat{<:Real})

Quantile-Quantile plot along with the confidence/credible interval of level `1-α`.

## Details

This function uses the Hessian matrix `H` provided in the argument.  
"""
function qqplotci(fd::DependentScalingModel, data::IDFdata, d::Real, H::PDMat{<:Real}, α::Real=.05)
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
        c = quantilecint(fd, data, d, pᵢ, H)
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
    qqplotci(fm::DependentScalingModel, data::IDFdata, d::Real, α::Real=.05)

Quantile-Quantile plot for the estimated distribution for duration d, along with the confidence interval of level `1-α`.
"""
function qqplotci(fd::DependentScalingModel, data::IDFdata, d::Real, α::Real=.05)
    
    H = IDFCurves.hessian(fd, data)
    
    return qqplotci(fd, data, d, H, α)
end

"""
    qqplotci(fm::MarginalScalingModel, α::Real=.05)

Quantile-Quantile plot along with the confidence/credible interval of level `1-α`.
"""
function qqplotci(fd::MarginalScalingModel, data::IDFdata, d::Real, α::Real=.05)
    
    return qqplotci(DependentScalingModel(fd, UncorrelatedStructure(), IdentityCopula), data, d, α)
end

"""
    qqplotci(fm::MarginalScalingModel, α::Real=.05, H::PDMat{<:Real})

Quantile-Quantile plot along with the confidence/credible interval of level `1-α`.

## Details

This function uses the Hessian matrix `H` provided in the argument.  
"""
function qqplotci(fd::MarginalScalingModel, data::IDFdata, d::Real, H::PDMat{<:Real}, α::Real=.05)
    return qqplotci(DependentScalingModel(fd, UncorrelatedStructure(), IdentityCopula), data, d, H, α)
end


"""
Les fonctions suivantes servent à représenter les courbes IDF à partir d'un modèle de scaling.
"""

"""
    get_durations_labels(durations::Vector{<:Real})

Returns a vector of labels (String) based on a vector of durations.
The returned vector contains the labels whiwh will appear on the x-axis wwhen plotting the IDF curve.

"""
function get_durations_labels(durations::Vector{<:Real})

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


"""
get_layers_IDFCurves(model::DependentScalingModel, T_values::Vector{<:Real}, durations_range::Vector{<:Real})

Returns the layers associated to the IDF curves based on the given mode, for each return period in T_values.
"""
function get_layers_IDFCurves(model::DependentScalingModel, T_values::Vector{<:Real}, durations_range::Vector{<:Real})

    data_return_levels = crossjoin( DataFrame(T = T_values),  DataFrame(d = durations_range) )
    transform!(data_return_levels, [:T, :d] => ((x,y) -> quantile.(Ref(model), y, 1 .- 1 ./ x)) => :return_level)

    layers = []
    for T in reverse(T_values)
        data = data_return_levels[data_return_levels[:,:T] .== T, :]
        push!(layers, layer(data, x = :d, y = :return_level, color = :T, Geom.line()))
    end

    return layers

end


"""
plotIDFCurves(model::DependentScalingModel; ...)

Pots the IDF curves based on the given model. Durations and return periods may be chosen by the user but have default values.
"""
function plotIDFCurves(model::DependentScalingModel; 
        T_values::Vector{<:Real}=[2,5,10,25,50,100],
        durations::Vector{<:Real}=[1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24],
        d_min::Union{Real, Nothing} = nothing,
        d_max::Union{Real, Nothing} = nothing,
        y_ticks::Union{Vector{<:Real}, Nothing} = nothing,
        palette::Union{Vector{<:Any}, Nothing} = nothing)

    if isnothing(d_min)
        d_min = minimum(durations)
    end
    if isnothing(d_max)
        d_max = maximum(durations)
    end
    durations = [ [d_min] ; durations[d_min .< durations .&& durations .< d_max] ; [d_max ]]

    T_values = sort(T_values)
    durations = sort(durations)
    
    d_step = d_min/10
    layers = get_layers_IDFCurves(model, T_values, collect(d_min:d_step:d_max))

    labels = get_durations_labels(durations)
    f_label(x) = labels[durations .≈ exp(x)][1]
    palette = [Scale.color_continuous().f((2*i-1)/(2*length(T_values))) for i in eachindex(T_values)]

    if isnothing(y_ticks)
        y_ticks = range(log(.9 * quantile(model, d_max, 1 - 1 / T_values[1])), log(1.1 * quantile(model, d_min, 1 - 1 / T_values[end])), 6)
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


"""
get_layers_pointwise_estimations(data::IDFdata, T_values::Vector{<:Real}, durations::Vector{<:Real}, show_confidence_intervals::Bool, ribbon::Bool)

Returns the layers associated to pointwise estimations of the return levels at the differents durations, for every return period in T_values.
"""
function get_layers_pointwise_estimations(data::IDFdata, T_values::Vector{<:Real}, durations::Vector{<:Real}, show_confidence_intervals::Bool, ribbon::Bool)

    if show_confidence_intervals
        data_return_levels = DataFrame(T = Float64[], d = Float64[], return_level = Float64[], conf_lower = Float64[], conf_upper = Float64[])
    else
        data_return_levels = DataFrame(T = Float64[], d = Float64[], return_level = Float64[])
    end
    
    for d in durations
        try
            fm = Extremes.gevfit(getdata(data, gettag(data,d)))

            for T in T_values
                if show_confidence_intervals
                    conf_int = cint(returnlevel(fm,T), .95)[1]
                    push!(data_return_levels, [T, d, quantile(fm, 1 - 1/T)[1], conf_int[1], conf_int[2]])
                else 
                    push!(data_return_levels, [T, d, quantile(fm, 1 - 1/T)[1]])
                end
            end
        catch e
            continue
        end
    end

    layers = []
    for T in reverse(T_values)
        data = data_return_levels[data_return_levels[:,:T] .== T, :]
        if show_confidence_intervals
            push!(layers, layer(data, x = :d, y = :return_level, ymin = :conf_lower, ymax = :conf_upper, color = :T, shape=[Shape.xcross], ribbon ? Geom.ribbon() : Geom.point(), Geom.errorbar(), Theme(alphas=[0.4])))
        else 
            push!(layers, layer(data, x = :d, y = :return_level, color = :T, shape=[Shape.xcross], Geom.point()))
        end
    end

    return layers

end

"""
get_layers_pointwise_estimations(model::DependentScalingModel, data::IDFdata, T_values::Vector{<:Real}, durations::Vector{<:Real}, show_confidence_intervals::Bool, ribbon::Bool)

Returns the layers associated to pointwise estimations of the return levels at the differents durations, for every return period in T_values.
"""
function get_layers_pointwise_estimations(model::DependentScalingModel, data::IDFdata, T_values::Vector{<:Real}, durations::Vector{<:Real}, show_confidence_intervals::Bool, ribbon::Bool, α::Real=.05)

    if show_confidence_intervals
        data_return_levels = DataFrame(T = Float64[], d = Float64[], return_level = Float64[], conf_lower = Float64[], conf_upper = Float64[])
    else
        data_return_levels = DataFrame(T = Float64[], d = Float64[], return_level = Float64[])
    end
    
    for d in durations
        try
            scaling_model = getmarginalmodel(model)

            for T in T_values
                p = 1-1/T
                q = quantile(scaling_model, d, p)
                if show_confidence_intervals
                    conf_int = quantilecint(scaling_model, data, d, p, α)
                    push!(data_return_levels, [T, d, q, conf_int[1], conf_int[2]])
                else 
                    push!(data_return_levels, [T, d, q])
                end
            end
        catch e
            continue
        end
    end

    layers = []
    for T in reverse(T_values)
        data = data_return_levels[data_return_levels[:,:T] .== T, :]
        if show_confidence_intervals
            push!(layers, layer(data, x = :d, y = :return_level, ymin = :conf_lower, ymax = :conf_upper, color = :T, shape=[Shape.xcross], ribbon ? Geom.ribbon() : Geom.point(), Geom.errorbar(), Theme(alphas=[0.4])))
        else 
            push!(layers, layer(data, x = :d, y = :return_level, color = :T, shape=[Shape.xcross], Geom.point()))
        end
    end

    return layers

end


"""
plotIDFCurves(model::DependentScalingModel, data::IDFdata; ...)

Plots the IDF curves based on the given model. Pointwise estimations are added (crosses) to illustrate the fitting.
Durations and return periods may be chosen by the user but have default values.
If show_confidence_intervals = true, 95% confidence intervals associated to the pointwise estimations are represented.
If ribbon = true, confidence intervals will be bounded by a ribbon above and below interval limits.
"""
function plotIDFCurves(model::DependentScalingModel, data::IDFdata; 
        show_confidence_intervals::Bool = false,
        ribbon::Bool = false,
        dgev_return_levels::Bool = false,
        α::Real=.05,
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

    T_values = sort(T_values)
    durations = sort(durations)
    
    d_step = d_min/10
    layers = get_layers_IDFCurves(model, T_values, collect(d_min:d_step:d_max))
    
    append!(layers, dgev_return_levels ? get_layers_pointwise_estimations(model, data, T_values, durations, show_confidence_intervals, ribbon, α) :  get_layers_pointwise_estimations(data, T_values, durations, show_confidence_intervals, ribbon))

    labels = get_durations_labels(durations)
    f_label(x) = labels[durations .≈ exp(x)][1]
    palette = [Scale.color_continuous().f((2*i-1)/(2*length(T_values))) for i in eachindex(T_values)]

    if isnothing(y_ticks)
        y_ticks = range(log(.9 * quantile(model, d_max, 1 - 1 / T_values[1])), log(1.1 * quantile(model, d_min, 1 - 1 / T_values[end])), 6)
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

function plotIDFCurves(marginalmodel::MarginalScalingModel; 
    T_values::Vector{<:Real}=[2,5,10,25,50,100],
    durations::Vector{<:Real}=[1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24],
    d_min::Union{Real, Nothing} = nothing,
    d_max::Union{Real, Nothing} = nothing,
    y_ticks::Union{Vector{<:Real}, Nothing} = nothing,
    palette::Union{Vector{<:Any}, Nothing} = nothing)

    return plotIDFCurves(DependentScalingModel(marginalmodel, UncorrelatedStructure(), IdentityCopula),
        T_values = T_values,
        durations = durations,
        d_min = d_min,
        d_max = d_max,
        y_ticks = y_ticks,
        palette = palette
    )

end

function plotIDFCurves(marginalmodel::MarginalScalingModel, data::IDFdata; 
    show_confidence_intervals::Bool = false,
    T_values::Vector{<:Real}=[2,5,10,25,50,100],
    durations::Vector{<:Real}=[1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24],
    d_min::Union{Real, Nothing} = nothing,
    d_max::Union{Real, Nothing} = nothing,
    y_ticks::Union{Vector{<:Real}, Nothing} = nothing)

    return plotIDFCurves(DependentScalingModel(marginalmodel, UncorrelatedStructure(), IdentityCopula), data,
        show_confidence_intervals = show_confidence_intervals,
        T_values = T_values,
        durations = durations,
        d_min = d_min,
        d_max = d_max,
        y_ticks = y_ticks
    )

end
