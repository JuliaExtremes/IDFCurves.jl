
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, Extremes, IDFCurves, LinearAlgebra
using Cairo, Gadfly, Fontconfig

using Test

## Application

# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Mtl_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Mtl_cdf_generalscaling.pdf"), fig)






# data at Vancouver
df = IDFCurves.dataset("1108446")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Van_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Mtl_cdf_generalscaling.pdf"), fig)







# data at Toronto
df = IDFCurves.dataset("6158731")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Tor_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Tor_cdf_generalscaling.pdf"), fig)





using CanadianClimateData

# data at Vancouver
df = CanadianClimateData.parse_idf_table("//Users/jalbert/Library/CloudStorage/Dropbox/Files/Data/ECCC/ECCC-idf_v3-30_2022_10_31/BC/idf_v3-30_2022_10_31_101_BC_1018621_VICTORIA_INTL_A.txt")

tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Tor_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

draw(PDF("Tor_cdf_generalscaling.pdf"), fig)


## Refactor fit_mle

df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)




"""
    initialize(::Type{<:SimpleScaling}, data::IDFdata, d₀::Real, lower_threshold::Real=0)

Construct an initial `SimpleScaling` model from IDF data.

Gumbel distributions are fitted independently at each duration using probability
weighted moments. The scaling exponent and reference-location parameter are then
initialized by a log-log regression of the fitted Gumbel locations on duration.
The reference-scale parameter is initialized from the fitted Gumbel scales using
the same scaling exponent.
"""
function initialize(::Type{<:SimpleScaling}, data::IDFdata, d₀::Real, lower_threshold::Real=0.0)

    d₀ > 0 || throw(ArgumentError("Reference duration must be positive, got d₀=$d₀"))

    d = Float64.(getduration.(data, gettag(data)))
    filter!(≥(lower_threshold), d)

    length(d) ≥ 2 || throw(ArgumentError("Lower threshold is too high, at least two durations are required to initialize SimpleScaling."))

    tags = gettag.(data, d)

    μ = Vector{Float64}(undef, length(tags))
    σ = Vector{Float64}(undef, length(tags))

    for (i, tag) in enumerate(tags)
        fd = fit(Gumbel, getdata(data, tag), method="pwm")

        μ[i] = location(fd)
        σ[i] = Distributions.scale(fd)
    end

    all(>(0), μ) || throw(ArgumentError("Fitted Gumbel locations must be positive to initialize SimpleScaling."))
    all(>(0), σ) || throw(ArgumentError("Fitted Gumbel scales must be positive to initialize SimpleScaling."))

    logd = log.(d ./ d₀)

    X = [ones(length(tags)) logd]
    β = X \ log.(μ)

    α = clamp(-β[2], 0.001, 0.999)

    μ₀ = exp(β[1])
    σ₀ = exp(mean(log.(σ) .+ α .* logd))

    return SimpleScaling(d₀, μ₀, σ₀, 0.0, α)

end

fm = initialize(SimpleScaling, data, 1., .25)

@testset "initialize(::SimpleScaling)" begin
            
    @test_throws ArgumentError initialize(SimpleScaling, data, -1.)
    @test_throws ArgumentError initialize(SimpleScaling, data, 1., 25.)

    fm = initialize(SimpleScaling, data, 1.)
    @test fm isa SimpleScaling

end











plotIDFCurves(fm, data)


function initial_duration_offset(fm::SimpleScaling, data::IDFdata, upper_treshold::Real=0.25)

    d = getduration.(data, gettag(data))
    filter!(<(upper_threshold), d)

    isempty(d) && throw(ArgumentError("Upper threshold is too low, no remaining durations for estimating δ."))

    tags = gettag.(data, d)

    α = IDFCurves.exponent(fm)
    μ₀ = location(fm)

    μ = Vector{Float64}(undef, length(tags))

    for (i, tag) in enumerate(tags)

        y = getdata(data, tag)
        fd = fit(Gumbel, y, method="pwm")

        μ[i] = location(fd)
    end

    z = -log.(μ ./ μ₀) ./ α

    δ = (d .- d₀) ./ expm1.(z) .- d₀

    δ̂ = median(δ)

    return δ̂
end

initial_duration_offset(fm, data, 1)

function initialize(pd::Type{<:GeneralScaling}, data::IDFdata, d₀::Real, threshold::Real=.25)

    (d₀ > 0) || throw(ArgumentError("Reference duration must be positive, got d₀=$d₀"))

    ss = initialize(SimpleScaling, data, d₀, threshold)

    α = IDFCurves.exponent(ss)
    μ₀ = IDFCurves.location(ss)
    σ₀ = IDFCurves.scale(ss)
    ξ = 0.

    δ = initial_duration_offset(ss, data, threshold)

    return GeneralScaling(d₀, μ₀, σ₀, ξ, α, δ)

end

fm = initialize(GeneralScaling, data, 1.)

plotIDFCurves(fm, data)