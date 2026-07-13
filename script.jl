
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, IDFCurves, LinearAlgebra, Test


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










train_data = excludeduration(data, "5min")

fm = fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(fm, 5/60)










"""
    excludeduration(data::IDFdata, tag_out::String)

Remove data of `data` corresponding to the duration identified by `tag_out`.
"""
function excludeduration(data::IDFdata, tag_out::String)

    tags = gettag(data)

    if tag_out ∉ tags
        throw(ArgumentError("tag_out not in the tags, got '$tag_out'."))
    end

    new_year = Dict{String, Vector{Int64}}()
    new_data = Dict{String, Vector{Float64}}()
    new_duration = Dict{String, Float64}()

    new_tag = setdiff(tags, [tag_out])

    for key in new_tag
        new_year[key] = getyear(data, key)
        new_data[key] = getdata(data, key)
        new_duration[key] = getduration(data, key)
    end

    return IDFdata(new_tag, new_duration, new_year, new_data)

end

nd = excludeduration(data, "5min")



    tags = ["30min", "1h", "24h"]
    durations = [0.5, 1, 24]
    years = [2020, 2021]
    y = hcat(1:2, 3:4, 5:6)

    d1 = Dict("30min" => 0.5, "1h" => 1., "24h" => 24.)
    d2 = Dict("30min" => years, "1h" => years, "24h" => years)
    d3 = Dict("30min" => y[:, 1], "1h" => y[:, 2], "24h" => y[:, 3])

    s = IDFdata(tags, d1, d2, d3)

    @testset "excludeduration(::IDFdata, d)" begin

        @test_throws ArgumentError excludeduration(s, "nonextistant_tag")

        s2 = excludeduration(s, "1h")
        @test getdata(s2) == Dict("30min" => y[:, 1], "24h" => y[:, 3])
        @test getduration(s2) == Dict("30min" => 0.5, "24h" => 24.)
        @test getyear(s2) == Dict("30min" => years, "24h" => years)
        @test gettag(s2) == ["30min", "24h"]

    end