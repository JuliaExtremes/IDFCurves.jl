
# julia --project=. --threads=auto application_universal_scaling.jl

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

simplescaling_pvalue = Vector{Float64}(undef, nstation)
generalscaling_pvalue = Vector{Float64}(undef, nstation)
universalscaling_pvalue = Vector{Float64}(undef, nstation)
nyear = Vector{Int64}(undef, nstation)

B = 999

Threads.@threads for i in eachindex(filenames)
# for i in eachindex(filenames)
    df = CSV.read(joinpath(@__FILE__, filepath, "canadian_stations_data", filenames[i]), DataFrame)
    data = IDFdata(df, "Year", duration_dict)

    nyear[i] = length(IDFCurves._common_years(data, tags))

    if nyear[i] < 15
        simplescaling_pvalue[i] = NaN
        generalscaling_pvalue[i] = NaN
        universalscaling_pvalue[i] = NaN
    else
        ss = IDFCurves.fit_mle(SimpleScaling, data, 1.)
        S = IDFCurves.validation_cvm_statistic(SimpleScaling, data, tag_out = "5min")
        
        Sstar = IDFCurves.scalingtest_bootstrap(ss, data, B=B)
        simplescaling_pvalue[i] = (1 + count(s -> s >= S, Sstar)) / (B + 1)

        if simplescaling_pvalue[i] < .05

            gs = IDFCurves.fit_mle(GeneralScaling, data, 1.)
            S = IDFCurves.validation_cvm_statistic(GeneralScaling, data, tag_out = "5min")
                
            Sstar = IDFCurves.scalingtest_bootstrap(gs, data, tag_out="5min", B=B)
            generalscaling_pvalue[i] = (1 + count(s -> s >= S, Sstar)) / (B + 1)

            if generalscaling_pvalue[i] < .05

                try
                    us = IDFCurves.fit_mle(UniversalScaling, data, 1.)
                    S = IDFCurves.validation_cvm_statistic(UniversalScaling, data, tag_out = "5min")
                        
                    Sstar = IDFCurves.scalingtest_bootstrap(us, data, tag_out="5min", B=B)
                    universalscaling_pvalue[i] = (1 + count(s -> s >= S, Sstar)) / (B + 1)
                catch
                    universalscaling_pvalue[i] = NaN
                end
            
            else
                universalscaling_pvalue[i] = generalscaling_pvalue[i]
            end
        else
            generalscaling_pvalue[i] = simplescaling_pvalue[i]
            universalscaling_pvalue[i] = simplescaling_pvalue[i]
        end
        
    end
    
    @info "Completed station" i

end

df_results.nyear = nyear
df_results.SimpleScaling_pvalue = simplescaling_pvalue
df_results.GeneralScaling_pvalue = generalscaling_pvalue
df_results.UniversalScaling_pvalue = universalscaling_pvalue


CSV.write(joinpath(@__FILE__, filepath, "scalingtest_canadian_stations.csv"), df_results)

## Display the p-value

using CSV, DataFrames, Distributions, Extremes, IDFCurves
using CanadaMap, CairoMakie

filepath = "/Users/jalbert/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC"
filename = joinpath(@__FILE__, filepath, "scalingtest_canadian_stations.csv")

df = CSV.read(filename, DataFrame)

p = clamp.(df.SimpleScaling_pvalue, 1e-10, 1.0)
values = -log10.(p)

fig, ga = generate_canada_map()

sc = scatter!(
    ga,
    df.Lon,
    df.Lat;
    color = values,
    colormap = :viridis,
    colorrange = (0, 3),
    markersize = 10
)

Colorbar(
    fig[1, 2],
    sc;
    label = "p-value",
    ticks = (
        [0, -log10(0.05), 2, 3],
        ["1", "0.05", "0.01", "≤0.001"]
    )
)

ga.xticklabelsvisible[] = false
ga.yticklabelsvisible[] = false

fig

save(joinpath(@__FILE__, filepath, "simple_scaling_pvalue.png"), fig; pt_per_unit = 1)


p = clamp.(df.GeneralScaling_pvalue, 1e-10, 1.0)
values = -log10.(p)

fig, ga = generate_canada_map()

sc = scatter!(
    ga,
    df.Lon,
    df.Lat;
    color = values,
    colormap = :viridis,
    colorrange = (0, 3),
    markersize = 10
)

Colorbar(
    fig[1, 2],
    sc;
    label = "p-value",
    ticks = (
        [0, -log10(0.05), 2, 3],
        ["1", "0.05", "0.01", "≤0.001"]
    )
)

ga.xticklabelsvisible[] = false
ga.yticklabelsvisible[] = false

fig

save(joinpath(@__FILE__, filepath, "general_scaling_pvalue.png"), fig; pt_per_unit = 1)




p = clamp.(df.UniversalScaling_pvalue, 1e-10, 1.0)
values = -log10.(p)

fig, ga = generate_canada_map()

sc = scatter!(
    ga,
    df.Lon,
    df.Lat;
    color = values,
    colormap = :viridis,
    colorrange = (0, 3),
    markersize = 10
)

Colorbar(
    fig[1, 2],
    sc;
    label = "p-value",
    ticks = (
        [0, -log10(0.05), 2, 3],
        ["1", "0.05", "0.01", "≤0.001"]
    )
)

ga.xticklabelsvisible[] = false
ga.yticklabelsvisible[] = false

fig

save(joinpath(@__FILE__, filepath, "universal_scaling_pvalue.png"), fig; pt_per_unit = 1)

count(p .< .05)/count(.!(isnan.(p)))


## Location of stations

fig, ga = generate_canada_map()

sc = scatter!(
    ga,
    df.Lon,
    df.Lat;
    color="#003262",
    markersize = 10
)

ga.xticklabelsvisible[] = false
ga.yticklabelsvisible[] = false

fig

save(joinpath(@__FILE__, filepath, "station_locations.png"), fig; pt_per_unit = 1)
