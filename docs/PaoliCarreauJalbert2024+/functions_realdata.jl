replace_negative_by_missing(x) = ismissing(x) ? x : ((x < 0) ? missing : x)

function preprocess(df::DataFrame, tags::Vector{<:AbstractString}, duration_dict::Dict)

    pp_df = copy(df)
    for name in tags

        # Transforming precip. accumulations into precip. intensities
        d = duration_dict[name]
        transform!(pp_df, Symbol(name) => (x -> x ./ d) => Symbol(name))

        # Replacing negative values by missing
        transform!(pp_df, Symbol(name) => (x -> replace_negative_by_missing.(x)) => Symbol(name))

    end

    return pp_df

end



function plotIDFCurves_regression(data::IDFdata,
    show_confidence_intervals::Bool = false,
    T_values::Vector{<:Real}=[2,5,10,25,50,100],
    durations::Vector{<:Real}=[1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24],
    d_min::Union{Real, Nothing} = nothing,
    d_max::Union{Real, Nothing} = nothing,
    y_ticks::Union{Vector{<:Real}, Nothing} = nothing)

    # pointwise estimations 
    pointwise_distribs = Dict()
    for d in durations
        pointwise_distribs[d] = Extremes.gevfit(getdata(data, gettag(data,d)))
    end
    function pointwise_quantile(d, T)
        return quantile(pointwise_distribs[d], 1 - 1/T)[1]
    end

    # layers with crosses
    layers = IDFCurves.get_layers_pointwise_estimations(data, T_values, durations, show_confidence_intervals)


    # layers with regression lines
    labels = IDFCurves.get_durations_labels(durations)
    f_label(x) = labels[durations .≈ exp(x)][1]
    palette = [Scale.color_continuous().f((2*i-1)/(2*length(T_values))) for i in eachindex(T_values)]

    if isnothing(y_ticks)
        y_ticks = [log(1),log(2),log(5),log(10),log(20),log(50),log(100),log(200), log(500)]
    end

    for T in reverse(T_values)
        data_return_levels = DataFrame(d = durations)
        transform!(data_return_levels, :d => ( x -> pointwise_quantile.(x, Ref(T))) => :return_level)
        transform!(data_return_levels, :d => ( x -> log.(x)) => :log_d)
        transform!(data_return_levels, :return_level => ( x -> log.(x)) => :log_return_level)
        
        reg_model = lm(@formula(log_return_level ~log_d), data_return_levels)
    
        data_return_levels[!,:reg_return_level] = exp.(predict(reg_model))
        data_return_levels[!,:T] .= T
        
        push!(layers, layer(data_return_levels, x = :d, y = :reg_return_level, color = :T, Geom.line()))
    end

    p = plot(layers..., Scale.x_log(labels = f_label), Scale.y_log(labels = y -> "$(round(exp(y)))"),
            Scale.color_discrete_manual(palette...),
            Guide.xticks(ticks = log.(durations)),
            Guide.yticks(ticks = y_ticks),
            Guide.xlabel("Rainfall duration"),
            Guide.ylabel("Rainfall intensity (mm/h)"),
            Guide.colorkey(title="Return period T (years)"),
            Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
                key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
            )


end





function plotCDFscomparison(pd_type::Type{<:MarginalScalingModel}, data::IDFdata, tag_out::String)

    d_out = getduration(data, tag_out)

    y_valid = getdata(data, tag_out)
    n = length(y_valid)
    F_n(x) = sum([y_valid[i] <= x for i in 1:n])/n

    train_data = IDFCurves.excludeduration(data, d_out)
    fitted_model = IDFCurves.fit_mle(pd_type, train_data, d_out)
    estim_distrib_valid = IDFCurves.getdistribution(fitted_model, d_out)


    l1 = layer(x -> cdf(estim_distrib_valid, x), 0, (floor(maximum(y_valid)/50) + 2) * 50, color = ["F_θ̂"])
    l2 = layer(F_n, 0, (floor(maximum(y_valid)/50) + 2) * 50, color = ["F̂_n"])
    p = plot(l1,l2,
        Guide.xlabel("Rainfall intensity"),
        Guide.ylabel("Probability"),
        Guide.colorkey(title = "Distribution function"),
        Theme(line_width = 1.5pt, point_size = 4pt, major_label_font_size = 15pt, 
                            key_label_font_size = 12pt, key_title_font_size  =15pt, minor_label_font_size = 12pt)
                )

end


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
    plt.title("Map of canadian stations and their respective scaling models", fontsize=15)

    # Stations
    ax.scatter(SS_stations.Lon, SS_stations.Lat, s=SS_stations.nyear, transform=ccrs.PlateCarree(),  c="blue", alpha=1., zorder=510, label = "Simple Scaling")
    ax.scatter(GS_stations.Lon, GS_stations.Lat, s=GS_stations.nyear, transform=ccrs.PlateCarree(),  c="red", alpha=1., zorder=510, label = "General Scaling")
    ax.scatter(NS_stations.Lon, NS_stations.Lat, s=NS_stations.nyear, transform=ccrs.PlateCarree(),  c="black", alpha=1., zorder=510, label = "No Scaling")

    ax.legend(loc="upper right", fontsize="x-large")

    # Enregistrement de la figure
    plt.savefig(filename, dpi=600);
    
    plt.show()

end