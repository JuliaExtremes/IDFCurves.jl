
# julia --project=. --threads=auto application.jl

# using Pkg
# pkg"activate ."

using CSV, DataFrames, Distributions, Extremes, IDFCurves

filepath = "/Users/jalbert/Library/CloudStorage/Dropbox/Files/Papers/InProgress/PaoliCarreauJalbert2024/JRSSC"
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




# import IDFCurves: scalingtype, _validation_tag, _common_years, _restrict_years, _pseudoobs_matrix, _idfdata_from_pseudoobs, excludeduration
# using Random

# function scalingtest_bootstrap(
#     fitted_model::MarginalScalingModel,
#     data::IDFdata,
#     initialmodel::MarginalScalingModel;
#     tag_out=nothing,
#     B::Integer=999,
#     rng=Random.default_rng(),
# )
#     B > 0 || throw(ArgumentError("B must be positive."))

#     scalingtype(initialmodel) === scalingtype(fitted_model) ||
#         throw(ArgumentError("Fitted model and initial model must be of the same type, got $(scalingtype(fitted_model)) ≠ $(scalingtype(initialmodel))"))

#     pd_type = scalingtype(fitted_model)

#     tags = gettag(data)
#     if tag_out === nothing
#         tag_out = _validation_tag(data)
#     else
#         tag_out = _validation_tag(data, tag_out)
#     end

#     # Restrict the bootstrap to complete years across all durations.
#     years = _common_years(data, tags)

#     length(years) > 0 || throw(ArgumentError(
#         "There is no common year across all durations.",
#     ))

#     data_common = _restrict_years(data, years; tags=tags)

#     # Empirical cross-duration dependence, represented by yearly rank vectors.
#     U = _pseudoobs_matrix(data_common, tags)
#     n = size(U, 1)

#     Sstar = Vector{Union{Float64,Missing}}(undef, B)

#     # Generate bootstrap indices sequentially to avoid sharing the RNG across threads.
#     bootstrap_indices = [rand(rng, 1:n, n) for _ in 1:B]

#     for b in 1:B
#         idx = bootstrap_indices[b]
#         Ustar = U[idx, :]

#         data_star = _idfdata_from_pseudoobs(data_common, fitted_model, Ustar)

#         try
#             Tstar = scalingtest(pd_type, data_star, initialmodel; tag_out=tag_out)
#             Sstar[b] = T.test_statistic
#         catch
#             Sstar[b] = missing
#         end
#     end

#     return Sstar
# end


# simplescaling_appropriate = falses(nstation)
# generalscaling_appropriate = falses(nstation)
# B = 3

# # Threads.@threads for i in eachindex(filenames)
#     for i in eachindex(filenames)

#     df = CSV.read(joinpath(@__FILE__, filepath, "canadian_stations_data", filenames[i]), DataFrame)
#     data = IDFdata(df, "Year", duration_dict)

#     ss = IDFCurves.fit_mle(SimpleScaling, data, 1.)
#     T = scalingtest(SimpleScaling, data, tag_out="5min")
#     S = T.test_statistic

#     train_data = excludeduration(data, "5min")
#     initialmodel = initialize(SimpleScaling, train_data, 1.0)

#     Sstar = scalingtest_bootstrap(ss, data, initialmodel, B=B)
#     filter!(!ismissing, Sstar)

#     adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / length(Sstar)

#     if adjusted_pvalue < 0.05 # SimpleScaling rejected
#         gs = IDFCurves.fit_mle(GeneralScaling, data, 1.)
#         T = scalingtest(GeneralScaling, data, tag_out="5min")
#         S = T.test_statistic

#         train_data = excludeduration(data, "5min")
#         initialmodel = initialize(GeneralScaling, train_data, 1.0)

#         Sstar = scalingtest_bootstrap(ss, data, initialmodel, B=B)
#         ilter!(!ismissing, Sstar)

#         adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / length(Sstar)

#         generalscaling_appropriate[i] = adjusted_pvalue > 0.05
#     else
#         simplescaling_appropriate[i] = true
#         generalscaling_appropriate[i] = true
#     end
#     println(i)

# end

# df_results.SimpleScaling = simplescaling_appropriate
# df_results.GeneralScaling = generalscaling_appropriate

# CSV.write(joinpath(@__FILE__, filepath, "scalingtest_canadian_stations.csv"), df_results)


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
        valid_computation[i] = false
    end
end

df_results.SimpleScaling = simplescaling_appropriate
df_results.GeneralScaling = generalscaling_appropriate
df_results.valid = valid_computation

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