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

using Distributions, ForwardDiff, LinearAlgebra

function quantilecint_test(fd::DependentScalingModel, data::IDFdata, duration::Real, p::Real, y::Real, α::Real=.05)
    q = quantile.(fd, duration, p, y)
    H = IDFCurves.hessian(fd, data)
    v = IDFCurves.quantilevar(fd, data, duration, p, H, y)

    dist = Normal(q, sqrt(v))
    ci = quantile.(dist, [α/2, 1-α/2])

    # Standardize the confidence intervals
    pd = getmarginalmodel(fd)
    marginal = getdistribution(pd, duration)[y]
    
    μ = location.(marginal)
    σ = scale.(marginal)
    ξ = shape.(marginal)

    standardized_ci = 1 ./ ξ * log.(1 .+ ξ ./ σ .* (ci .- μ))

    return standardized_ci    
end

function qqplot_std_data(fd::DependentScalingModel, data::IDFdata, durations::AbstractArray{<:Real})::Plot

    z_all = Float64[]
    # z_inf = Float64[]
    # z_sup = Float64[]

    # Iterate over each duration and standardize the data
    for d in durations
        z = standardize(getmarginalmodel(fd), data, d)

        # Calculate confidence intervals
        # Confidence intervals dont work yet, they don't seem to cover model quantiles
        # tag = gettag(data, d)
        # y = getdata(data, tag)
        # n = length(y)
        # p = (1:n) ./ (n+1)
        # for (i, pᵢ) in enumerate(p)
        #     c = quantilecint_test(fd, data, d, pᵢ, i)

        #     append!(z_inf, c[1])
        #     append!(z_sup, c[2])
        # end

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

    # Append confidence intervals to the DataFrame
    df.LowerBound = lower_bound[:]
    df.UpperBound = upper_bound[:]

    # df[:,:Inf] = z_inf
    # df[:,:Sup] = z_sup
    

    l1 = layer(df, x=:Model, y=:Empirical, Geom.point, Geom.abline(color="black", style=:dash), Theme(default_color="black", discrete_highlight_color=c->nothing))
    # Add a ribbon for the confidence intervals
    l2 = layer(df, x=:Model, ymin=:LowerBound, ymax=:UpperBound, Geom.ribbon, Theme(default_color="lightgray"))

    # l2 = layer(df, x=:Model, ymin=:Inf, ymax=:Sup, Geom.ribbon, Theme(lowlight_color=c->"lightgray"))
    p = plot(l1, l2, Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Residual Quantile Plot for durations: $durations"), Theme(background_color="white"))
    p.scales = [
        Scale.x_continuous(minvalue=-2, maxvalue=6),
        Scale.y_continuous(minvalue=-5, maxvalue=10)
    ]

    return p
end