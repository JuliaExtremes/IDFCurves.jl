
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
q, _ = Extremes.ecdf(getdata(data, "5min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
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

count(df.SimpleScaling)
count(df.GeneralScaling)

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

ccrs = pyimport("cartopy.crs")
cfeat = pyimport("cartopy.feature")
cticker = pyimport("cartopy.mpl.ticker")
gliner = pyimport("cartopy.mpl.gridliner")
mpl = pyimport("mpl_toolkits.axes_grid1.inset_locator")
plt = pyimport("matplotlib.pyplot")
#fig = plt.figure(figsize=(13, 7), constrained_layout=true)

mpl_ticker = pyimport("matplotlib.ticker")
mpl_colors = pyimport("matplotlib.colors")


function draw_map(SS_stations::DataFrame, GS_stations::DataFrame, NS_stations::DataFrame, filename::String)

    fig = plt.figure(figsize=(13, 7), constrained_layout=true)
    
    central_longitude = -(91 + 52 / 60)

    # Création de la carte
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=central_longitude))

    # Définition des limites 
    xlims = (-145, -50)
    ylims = (38, 79)
    ax.set_extent([xlims[1], xlims[2], ylims[1], ylims[2]])

    # # Grille
    gl = ax.gridlines(draw_labels=false, lw=1., zorder=12, color="gray", alpha=0.3, linestyle="--")

    ## Ajout des features :

    # Frontières politiques
    country_bord = cfeat.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="50m",
        facecolor="none")

    ax.add_feature(country_bord, edgecolor="gray", zorder=10)

    # Provinces
    states_provinces = cfeat.NaturalEarthFeature(category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none")

    ax.add_feature(states_provinces, edgecolor="gray", zorder=10)

    # Terre
    land = cfeat.NaturalEarthFeature(
        category="physical",
        name="land",
        scale="50m",
        edgecolor="k",
        facecolor=cfeat.COLORS["land"])

    ax.add_feature(land, zorder=4)

    # Ocean/mer
    ocean = cfeat.NaturalEarthFeature(
        category="physical",
        name="ocean",
        scale="50m",
        edgecolor="none",
        facecolor=cfeat.COLORS["water"])

    ax.add_feature(ocean)

    # Lacs
    lakes = cfeat.NaturalEarthFeature(
        category="physical",
        name="lakes",
        #scale="10m",
        scale="50m",
        #scale="110m",
        edgecolor=cfeat.COLORS["water"],
        facecolor=cfeat.COLORS["water"])

    ax.add_feature(lakes, zorder=5)

    # Rivières
    rivers = cfeat.NaturalEarthFeature(
        category="physical",
        name="rivers_lake_centerlines",
        #scale="10m",
        scale="50m",
        edgecolor=cfeat.COLORS["water"],
        facecolor="none")

    ax.add_feature(rivers, zorder=6)

    # Define the xticks for longitude
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    # Titre
    # plt.title("Map of canadian stations and their respective scaling models", fontsize=15)

    # Stations
    ax.scatter(SS_stations.Lon, SS_stations.Lat, s=SS_stations.nyear, transform=ccrs.PlateCarree(),  c="blue", alpha=1., zorder=510, label = "Simple Scaling")
    ax.scatter(GS_stations.Lon, GS_stations.Lat, s=GS_stations.nyear, transform=ccrs.PlateCarree(),  c="red", alpha=1., zorder=510, label = "General Scaling")
    ax.scatter(NS_stations.Lon, NS_stations.Lat, s=NS_stations.nyear, transform=ccrs.PlateCarree(),  c="black", alpha=1., zorder=510, label = "No Scaling")

    ax.legend(loc="upper right", fontsize="x-large")

    # Enregistrement de la figure
    plt.savefig(filename, dpi=600);
    
    plt.show()

end

SS_stations = filter(row -> row.SimpleScaling, df)
GS_stations = filter(row -> row.GeneralScaling && !row.SimpleScaling, df )
NS_stations = filter(row -> !row.GeneralScaling, df )

draw_map(SS_stations, GS_stations, NS_stations, "canadian_stations_map.png")


## Show simulation results

using Pkg
pkg"activate ."

using CSV, DataFrames, IDFCurves
using Cairo, Gadfly, Fontconfig

using CategoricalArrays

df = CSV.read("SimpleScaling_type1_error.csv", DataFrame)

df.ξ = categorical(string.(df.ξ))

plot(df, x=:n, y=:RejectionRate, color=:ξ, Geom.line, Geom.point)

df = CSV.read("GeneralScaling_type1_error.csv", DataFrame)

df.ξ = categorical(string.(df.ξ))

plot(df, x=:n, y=:RejectionRate, color=:ξ, Geom.line, Geom.point)

