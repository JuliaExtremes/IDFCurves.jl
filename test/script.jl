using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, ForwardDiff, Gadfly,  LogExpFunctions, Optim

import IDFCurves.IDFdata
import Distributions: fit_mle, params, quantile
import Extremes: fit_mle, qqplot



df = CSV.read(joinpath("data","IDF_702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])
# fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [1, 1, 0, .9, 1])

m = getdistribution(fd, 24)
y = getdata(data, "24h")

# IDFCurves.qqplot(m, y)




quantile(fd, 1, .9)