using Pkg
pkg"activate ."

using Distributions, IDFCurves, LinearAlgebra, Test

# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

@time scalingtest(SimpleScaling, data, tag_out = "5min", q=100)
@time scalingtest(GeneralScaling, data, tag_out = "5min", q=100) 