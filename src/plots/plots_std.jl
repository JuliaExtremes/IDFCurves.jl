"""
    ecdf(y::Vector{<:Real})::Tuple{Vector{<:Real}, Vector{<:Real}}

Compute the empirical cumulative distribution function using the Gumbel formula.

The empirical quantiles are computed using the Gumbel plotting positions as
as recommended by [Makkonen (2006)](https://journals.ametsoc.org/jamc/article/45/2/334/12668/Plotting-Positions-in-Extreme-Value-Analysis).

# Example
```julia-repl
julia> (x, F̂) = Extremes.ecdf(y)
```

# Reference
Makkonen, L. (2006). Plotting positions in extreme value analysis. Journal of
Applied Meteorology and Climatology, 45(2), 334-340.
"""
function ecdf(y::AbstractArray{<:Real})::Tuple{AbstractArray{<:Real}, AbstractArray{<:Real}}
    ys = sort(y)
    n = length(ys)
    p = collect(1:n)/(n+1)

    return ys, p
end

function standardize(y::Real, μ::Real, σ::Real, ξ::Real)::Real

    if ξ ≈ 0
        z = (y-μ)/σ
    else
        z = 1 / ξ * log( 1 + ξ/σ * ( y - μ ) )
    end

    return z

end

function standardize(pd::MarginalScalingModel, data::IDFdata, d::Real)::AbstractArray{<:Real}

    marginal = getdistribution(pd, d)
    tag = gettag(data, d)
    y = getdata(data, tag)

    return standardize.(y, location.(marginal), scale.(marginal), shape.(marginal))

end

function qqplot_std_data(
    fd::DependentScalingModel, 
    data::IDFdata, 
    durations::AbstractArray{<:Real}, 
    title::String="Residual Quantile Plot", 
    xaxis_title::String="Model", 
    yaxis_title::String="Empirical", 
    axis_scales::AbstractArray{<:Gadfly.Scale.ContinuousScale}=[
        Scale.x_continuous(minvalue=-2, maxvalue=8),
        Scale.y_continuous(minvalue=-2, maxvalue=5)
    ])::Plot
    # Iterate over each duration and standardize the data
    z_all = Float64[]
    for d in durations
        z = standardize(getmarginalmodel(fd), data, d)
        append!(z_all, z)
    end

    y, p = ecdf(z_all)

    df = DataFrame(Model = quantile.(Gumbel(), p), Empirical = y)

    n_samples = 10000
    n = length(z_all)
    simulated_quantiles = zeros(Float64, n_samples, n)


    # Bootstrap simulation: generate quantiles from the Gumbel distribution
    for i in 1:n_samples
        simulated_data = rand(Gumbel(), n)  # Simulate data from the Gumbel distribution
        simulated_quantiles[i, :] = sort(simulated_data)
    end

    # Compute the 5th and 95th percentile confidence intervals for each quantile
    lower_bound = mapslices(x -> quantile(x, 0.025), simulated_quantiles, dims=1)
    upper_bound = mapslices(x -> quantile(x, 0.975), simulated_quantiles, dims=1)

    df.LowerBound = lower_bound[:]
    df.UpperBound = upper_bound[:]
    

    l1 = layer(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="black", style=:dash), Theme(default_color="black", discrete_highlight_color=c->nothing))
    # Add a ribbon for the confidence intervals
    l2 = layer(df, x=:Model, ymin=:LowerBound, ymax=:UpperBound, Geom.ribbon, Theme(default_color="lightgray"))
    p = plot(l1, l2, Guide.xlabel(xaxis_title), Guide.ylabel(yaxis_title), Guide.title(title), Theme(
            line_width = 1.5pt, point_size = 4pt, major_label_font_size = 20pt, key_label_font_size = 20pt, 
            key_title_font_size = 15pt, minor_label_font_size = 20pt, background_color="white"
        )
    )
    p.scales = axis_scales

    return p
end