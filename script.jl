
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, Extremes, IDFCurves, LinearAlgebra, Optim
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

# draw(PDF("Mtl_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Mtl_cdf_generalscaling.pdf"), fig)


# data at Nanaimo
df = IDFCurves.dataset("1025369")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

plotIDFCurves(fd, data)

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


train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Nan_cdf_simplescaling.pdf"), fig)




# data at Vancouver (Cette station nécessite le universalscaling !)
df = IDFCurves.dataset("1108446")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out = "10min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data; tag_out = "10min", B = B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)

# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="10min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data; tag_out = "10min", B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "10min")
q, _ = Extremes.ecdf(getdata(data, "10min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 10/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("10-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Van_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Mtl_cdf_generalscaling.pdf"), fig)



