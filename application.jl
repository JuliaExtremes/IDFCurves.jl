
# julia --project=. --threads=auto application.jl

# using Pkg
# pkg"activate ."

using CSV, DataFrames, Distributions, Extremes, IDFCurves

filepath = "/Users/jalbert/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC"
filenames = filter(f -> endswith(lowercase(f), ".csv"), readdir(joinpath(@__DIR__, filepath, "canadian_stations_data")))

nstation = length(filenames)

station_list = CSV.read(joinpath(@__FILE__, filepath, "_station_list.csv"), DataFrame)

df_results = DataFrame(Name=String[], ID=String[], Lat=Float64[], Lon=Float64[], Elevelation=Float64[])

for (i, filename) in enumerate(filenames)
    ID, _ = splitext(filename)
    ind = findfirst(station_list.ID .== ID)
    push!(df_results, station_list[ind, :], cols=:subset)
end

tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))


# Ici on est trop sévère sur le rejet, il suffit d'un replicat qui ne converge pas pour rejeter l'hypothèse nulle

simplescaling_appropriate = falses(nstation)
generalscaling_appropriate = falses(nstation)
valid_computation = trues(nstation)

B = 999

Threads.@threads for i in eachindex(filenames)

    df = CSV.read(joinpath(@__FILE__, filepath, "canadian_stations_data", filenames[i]), DataFrame)
    data = IDFdata(df, "Year", duration_dict)

    try
        ss = IDFCurves.fit_mle(SimpleScaling, data, 1.)
        T = scalingtest(SimpleScaling, data, tag_out="5min")
        S = T.test_statistic

        Tstar = IDFCurves.scalingtest_bootstrap(ss, data, B=B)
        Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
        adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)

        if adjusted_pvalue < .05 # SimpleScaling rejected
            gs = IDFCurves.fit_mle(GeneralScaling, data, 1.)
            T = scalingtest(GeneralScaling, data, tag_out="24h")
            S = T.test_statistic

            Tstar = IDFCurves.scalingtest_bootstrap(gs, data, tag_out="24h", B=B)
            Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
            adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)

            generalscaling_appropriate[i] = adjusted_pvalue >.05
        else
            simplescaling_appropriate[i] = true
            generalscaling_appropriate[i] = true
        end
        println(i)
    catch
        valid_computation[i] = false
    end
end

df_results.SimpleScaling = simplescaling_appropriate
df_results.GeneralScaling = generalscaling_appropriate
df_results.valid = valid_computation

CSV.write(joinpath(@__FILE__, filepath, "scalingtest_canadian_stations.csv"), df_results)





## Refactored application.jl


using LinearAlgebra
using Random

BLAS.set_num_threads(1)

simple_ok = falses(nstation)
general_ok = falses(nstation)
simple_pvalue = fill(NaN, nstation)
general_pvalue = fill(NaN, nstation)
valid = trues(nstation)

seeds = rand(Xoshiro(2026), UInt64, nstation)

function assess_station(
    data::IDFdata;
    B::Integer=999,
    level::Real=0.05,
    rng=Random.default_rng(),
)

    # Simple Scaling
    ss_full = fit_mle(SimpleScaling, data, 1.0)

    ss_result = scalingtest_bootstrap_pvalue(
        ss_full,
        data;
        tag_out="5min",
        B=B,
        rng=rng,
    )

    if ss_result.pvalue >= level
        return (
            SimpleScaling=true,
            GeneralScaling=true,
            SimpleScalingPValue=ss_result.pvalue,
            GeneralScalingPValue=NaN,
        )
    end

    # General Scaling is evaluated only if Simple Scaling is rejected.
    gs_full = fit_mle(GeneralScaling, data, 1.0)

    gs_result = scalingtest_bootstrap_pvalue(
        gs_full,
        data;
        tag_out="24h",
        B=B,
        rng=rng,
    )

    return (
        SimpleScaling=false,
        GeneralScaling=gs_result.pvalue >= level,
        SimpleScalingPValue=ss_result.pvalue,
        GeneralScalingPValue=gs_result.pvalue,
    )
end

Threads.@threads :dynamic for i in eachindex(filenames)

    try
        file = joinpath(
            filepath,
            "canadian_stations_data",
            filenames[i],
        )

        station_df = CSV.read(file, DataFrame)
        data = IDFdata(station_df, "Year", duration_dict)

        result = assess_station(
            data;
            B=B,
            level=0.05,
            rng=Xoshiro(seeds[i]),
        )

        simple_ok[i] = result.SimpleScaling
        general_ok[i] = result.GeneralScaling
        simple_pvalue[i] = result.SimpleScalingPValue
        general_pvalue[i] = result.GeneralScalingPValue

        @info "Completed station" i filename=filenames[i]

    catch err
        valid[i] = false

        @warn(
            "Station computation failed",
            station=filenames[i],
            exception=(err, catch_backtrace()),
        )
    end
end
