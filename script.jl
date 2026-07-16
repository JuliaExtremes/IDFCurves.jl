
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

##

using Pkg
pkg"activate ."

using DataFrames, Distributions, Extremes, IDFCurves, LinearAlgebra, Optim
using Cairo, Gadfly, Fontconfig

using Test

# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fm = IDFCurves.fit_mle(SimpleScaling, data, 1.)


using ForwardDiff, PDMats
import IDFCurves.scalingtype

function hessian(fm::MarginalScalingModel, data::IDFdata)

    T = scalingtype(fm)
    d₀ = duration(fm)
    θ̂ = collect(params(fm))

    model(θ::DenseVector{<:Real}) = T(d₀, θ...)
    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    H = ForwardDiff.hessian(fobj, θ̂)

    return PDMat(Symmetric(H))

end

hessian(fm, data)

@test hessian(fm, data) ≈ [24.2687 -12.2383 49.9538 -66.4114;
                -12.2383 41.7471 17.8326 -56.9225;
                49.9538 17.8326 1364.59 695.963;
                -66.4114 -56.9225 695.963 25166.9] rtol=0.05