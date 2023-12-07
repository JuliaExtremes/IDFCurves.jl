using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly



df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])

gettag(data)