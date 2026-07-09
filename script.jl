
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, IDFCurves, LinearAlgebra


# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out = "5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 100 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out = "5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 100 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


