
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, Extremes, IDFCurves
using Cairo, Gadfly, Fontconfig

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
Sstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
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
Sstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)

train_data = IDFCurves.excludeduration(data, "5min")
ss_model = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

distribution_5min = getdistribution(ss, 5/60)

y = getdata(data, "5min")
F(x::Real) = IDFCurves.ecdf_fun(y, x)


fig = plot([x->cdf(distribution_5min, x), x->F(x)], 0, 250,
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

# Gadfly.set_default_plot_size(7inch, 5inch)
fig = plotIDFCurves(fd, data)

# draw(PDF("IDFCurves_Nanaimo_simplescaling.pdf"), fig)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Sstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
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

draw(PDF("Nan_cdf_simplescaling.pdf"), fig)


## All canadian stations 

using CSV, DataFrames, Distributions, Plots

df = CSV.read("/Users/jalbert/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC/scalingtest_canadian_stations.csv", DataFrame)

filter!(row->row.nyear>15, df)

scatter(df.Lon, df.Lat, df.SimpleScaling_pvalue)

nstation = nrow(df)
nreject = count(df.SimpleScaling_pvalue .< .05)

p = nreject/nstation
q = 1-p

pd = Normal(p, sqrt(p*q/nstation))
quantile(pd, [.025, .95])

nreject = count(df.GeneralScaling_pvalue .< .05)

scatter(df.Lon, df.Lat, df.GeneralScaling_pvalue)

p = nreject/nstation
q = 1-p

pd = Normal(p, sqrt(p*q/nstation))
quantile(pd, [.025, .95])

using PyCall

df = CSV.read("/Users/jalbert/Library/CloudStorage/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC/scalingtest_canadian_stations.csv", DataFrame)

filepath = "/Users/jalbert/Library/CloudStorage/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC"
filenames = filter(f -> endswith(lowercase(f), ".csv"), readdir(joinpath(@__DIR__, filepath, "canadian_stations_data")))

nstation = length(filenames)

nyear = Vector{Int64}(undef, nstation)

for (i, filename) in enumerate(filenames)
    df_station = CSV.read(joinpath(@__FILE__,filepath, "canadian_stations_data",filename), DataFrame)
    nyear[i] = nrow(df_station)
end

df.nyear = nyear

count(df.SimpleScaling)
count(df.GeneralScaling)


## Show simulation results

using Pkg
pkg"activate ."

using CSV, DataFrames, IDFCurves
using Cairo, Gadfly, Fontconfig

using CategoricalArrays

folderpath = joinpath("/Users/jalbert/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC")

df = CSV.read("Simulations/simulation_results/SimpleScaling_type1_error.csv", DataFrame)

df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:n, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
Guide.ylabel("Rejection Rate"),
Coord.Cartesian(ymin=.04, ymax=.06)
)

filename = joinpath(@__FILE__, folderpath, "simplescaling_errorI.pdf")
draw(PDF(filename), fig)

df = CSV.read("Simulations/simulation_results/GeneralScaling_type1_error.csv", DataFrame)

df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:n, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"))

filename = joinpath(@__FILE__, folderpath, "generalscaling_errorI.pdf")
draw(PDF(filename), fig)


df = CSV.read("Simulations/simulation_results/SimpleScaling_power.csv", DataFrame)

df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:δ, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"))

filename = joinpath(@__FILE__, folderpath, "simplescaling_power.pdf")
draw(PDF(filename), fig)

df = CSV.read("Simulations/simulation_results/GeneralScaling_power.csv", DataFrame)

df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:α₁, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"))

filename = joinpath(@__FILE__, folderpath, "generalscaling_power.pdf")
draw(PDF(filename), fig)