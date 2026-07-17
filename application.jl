
# julia --project=. --threads=auto application.jl

# using Pkg
# pkg"activate ."

using CSV, DataFrames, Distributions, Extremes, IDFCurves

filepath = "/Users/jalbert/Library/CloudStorage/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC"
filenames = filter(f -> endswith(lowercase(f), ".csv"), readdir(joinpath(@__DIR__, filepath,"canadian_stations_data")))

nstation = length(filenames)

station_list = CSV.read(joinpath(@__FILE__,filepath,"_station_list.csv"), DataFrame)

df_results = DataFrame(Name = String[], ID = String[], Lat = Float64[], Lon = Float64[], Elevelation = Float64[])

for (i, filename) in enumerate(filenames)
    ID, _ = splitext(filename)
    ind = findfirst(station_list.ID .== ID)
    push!(df_results, station_list[ind, :], cols = :subset)
end


tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))

simplescaling_appropriate = falses(nstation)
generalscaling_appropriate = falses(nstation)
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
            T = scalingtest(GeneralScaling, data, tag_out="5min")
            S = T.test_statistic

            Tstar = IDFCurves.scalingtest_bootstrap(gs, data, B=B)
            Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
            adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)
            
            generalscaling_appropriate[i] = adjusted_pvalue >.05
        else
            simplescaling_appropriate[i] = true
            generalscaling_appropriate[i] = true
        end
        println(i)
    catch
    end
end

df_results.SimpleScaling = simplescaling_appropriate
df_results.GeneralScaling = generalscaling_appropriate

CSV.write(joinpath(@__FILE__, filepath, "scalingtest_canadian_stations.csv"), df_results)


## Wihtout dependence correction

# simplescaling_appropriate = falses(nstation)
# generalscaling_appropriate = falses(nstation)

# for i in eachindex(filenames)

#     df = CSV.read(joinpath(@__FILE__, filepath, filenames[i]), DataFrame)
#     data = IDFdata(df, "Year", duration_dict)

#     try
#         ss = IDFCurves.fit_mle(SimpleScaling, data, 1.)
#         T = scalingtest(SimpleScaling, data, tag_out="5min")
#         if IDFCurves.decision(T) # SimpleScaling rejected
#             gs = IDFCurves.fit_mle(GeneralScaling, data, 1.)
#             T = scalingtest(GeneralScaling, data, tag_out="5min")
#             generalscaling_appropriate[i] = !IDFCurves.decision(T)
#         else
#             simplescaling_appropriate[i] = true
#             generalscaling_appropriate[i] = true
#             println(i)
#         end
#     catch
#     end
# end