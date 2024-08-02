var documenterSearchIndex = {"docs":
[{"location":"functions/#Types-and-Functions","page":"Types and Functions","title":"Types and Functions","text":"","category":"section"},{"location":"functions/#Types","page":"Types and Functions","title":"Types","text":"","category":"section"},{"location":"functions/","page":"Types and Functions","title":"Types and Functions","text":"Modules = [IDFCurves]\nPrivate = false\nOrder = [:type]","category":"page"},{"location":"functions/#IDFCurves.GeneralScaling","page":"Types and Functions","title":"IDFCurves.GeneralScaling","text":"GeneralScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real, δ::Real)\n\nConstruct a GeneralScaling distribution type.\n\nDetails\n\nTODO\n\nmu_d = mu_0 left(fracd+deltad_0+delta right) ^-alpha qquad    sigma_d = sigma_0 left(fracd+deltad_0+delta right) ^-alpha qquad    xi_d = xi\n\nReferences\n\nKoutsoyiannis, D., Kozonis, D. and Manetas, A. (1998).  A mathematical framework for studying rainfall intensity-duration-frequency relationships, Journal of Hydrology, 206(1-2), 118-135, https://doi.org/10.1016/S0022-1694(98)00097-3.\n\n\n\n\n\n","category":"type"},{"location":"functions/#IDFCurves.IDFdata-Tuple{DataFrames.DataFrame, String, Dict{String, T} where T<:Real}","page":"Types and Functions","title":"IDFCurves.IDFdata","text":"IDFdata(df::DataFrame, year_id::String, duration::Dict{String, T} where T<:Real)\n\nConstruct a IDFdata structure from a DataFrame.\n\nDetails\n\nyear_id: string indicating the year column\nduration: Dictionary mapping the dataframe id to duration.\n\nSee the tutorial for an example.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.SimpleScaling","page":"Types and Functions","title":"IDFCurves.SimpleScaling","text":"SimpleScaling(d₀::Real, μ₀::Real, σ₀::Real, ξ::Real, α::Real)\n\nConstruct a simple scaling distribution type.\n\nDetails\n\nTODO\n\nmu_d = mu_0 left(fracdd_0 right) ^-alpha qquad    sigma_d = sigma_0 left(fracdd_0 right) ^-alpha qquad    xi_d = xi\n\nReferences\n\nKoutsoyiannis, D., Kozonis, D. and Manetas, A. (1998).  A mathematical framework for studying rainfall intensity-duration-frequency relationships, Journal of Hydrology, 206(1-2), 118-135, https://doi.org/10.1016/S0022-1694(98)00097-3.\n\n\n\n\n\n","category":"type"},{"location":"functions/#Functions","page":"Types and Functions","title":"Functions","text":"","category":"section"},{"location":"functions/","page":"Types and Functions","title":"Types and Functions","text":"Modules = [IDFCurves]\nPrivate = true\nOrder = [:function]\nPages = [\n    \"src/data.jl\",\n    \"src/covariance.jl\",\n    \"src/structures/idfdata.jl\",\n    \"src/structures/AbstractScalingModel/abstractscalingmodel.jl\",\n    \"src/structures/AbstractScalingModel/dGEV.jl\",\n    \"src/plots.jl\"\n    ]","category":"page"},{"location":"functions/#IDFCurves.dataset-Tuple{String}","page":"Types and Functions","title":"IDFCurves.dataset","text":"dataset(name::String)::DataFrame\n\nLoad the dataset associated with name.\n\nDetails\n\nSome datasets are available using the following names:\n\n1108446: short duration rainfall intensity-duration-frequency data recorded at  Vancouver Harbour CS\n6158731: short duration rainfall intensity-duration-frequency data recorded at the Toronto Leaster B. Pearson Internation Airport\n702S006: short duration rainfall intensity-duration-frequency data recorded at the Montreal Pierre-Elliott-Trudeau Internation Airport\n\nThese datasets have been retrieved from the Environment and Climate Change Canada website.\n\nExamples\n\njulia> IDFCurves.dataset(\"702S006\")\n\n\n\n\n\n","category":"method"},{"location":"functions/#Base.show-Tuple{IO, IDFdata}","page":"Types and Functions","title":"Base.show","text":"Base.show(io::IO, obj::IDFdata)\n\nOverride of the show function for the objects of type IDFdata.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.excludeduration-Tuple{IDFdata, Real}","page":"Types and Functions","title":"IDFCurves.excludeduration","text":"excludeduration(data::IDFdata, d::Real)\n\nRemove the data of data corresponding to the duration d.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getKendalldata-Tuple{IDFdata}","page":"Types and Functions","title":"IDFCurves.getKendalldata","text":"getKendalldata(obj::IDFdata)\n\nComputes the Kendall tau for each pair of durations for which obj contains data,     and returns them in a DataFrame.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getdata-Tuple{IDFdata, String, Int64}","page":"Types and Functions","title":"IDFCurves.getdata","text":"getdata(s::IDFdata, tag::String, year::Int)\n\nReturn the data corresponding to the duration tag tag and the year year.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getdata-Tuple{IDFdata, String}","page":"Types and Functions","title":"IDFCurves.getdata","text":"getdata(s::IDFdata, tag::String)\n\nReturn the data vector corresponding to the duration tag tag.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getdata-Tuple{IDFdata}","page":"Types and Functions","title":"IDFCurves.getdata","text":"getdata(s::IDFdata)\n\nReturn a dictionary containing the data vector for each duration tag.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getduration-Tuple{IDFdata, String}","page":"Types and Functions","title":"IDFCurves.getduration","text":"getduration(s::IDFdata, tag::String)\n\nReturn the vector of durations for the duration tag tag.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getduration-Tuple{IDFdata}","page":"Types and Functions","title":"IDFCurves.getduration","text":"getduration(s::IDFdata)\n\nReturn a dictionary containing the duration vector for each duration tag.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.gettag-Tuple{IDFdata, Real}","page":"Types and Functions","title":"IDFCurves.gettag","text":"gettag(data::IDFdata, d::Real)\n\nReturn the tag corresponding to the duration d if it exists; throw an error otherwise.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.gettag-Tuple{IDFdata}","page":"Types and Functions","title":"IDFCurves.gettag","text":"gettag(data::IDFdata)\n\nReturn the tag list.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getyear-Tuple{IDFdata, String}","page":"Types and Functions","title":"IDFCurves.getyear","text":"getyear(s::IDFdata, tag::String)\n\nReturn the vector of years for the duration tag tag.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.getyear-Tuple{IDFdata}","page":"Types and Functions","title":"IDFCurves.getyear","text":"getyear(s::IDFdata)\n\nReturn a dictionary containing the year vector for each duration tag.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.get_durations_labels-Tuple{Vector{<:Real}}","page":"Types and Functions","title":"IDFCurves.get_durations_labels","text":"get_durations_labels(durations::Vector{<:Real})\n\nReturns a vector of labels (String) based on a vector of durations. The returned vector contains the labels whiwh will appear on the x-axis wwhen plotting the IDF curve.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.get_layers_IDFCurves-Tuple{DependentScalingModel, Vector{<:Real}, Vector{<:Real}}","page":"Types and Functions","title":"IDFCurves.get_layers_IDFCurves","text":"getlayersIDFCurves(model::DependentScalingModel, Tvalues::Vector{<:Real}, durationsrange::Vector{<:Real})\n\nReturns the layers associated to the IDF curves based on the given mode, for each return period in T_values.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.get_layers_pointwise_estimations-Tuple{IDFdata, Vector{<:Real}, Vector{<:Real}, Bool}","page":"Types and Functions","title":"IDFCurves.get_layers_pointwise_estimations","text":"getlayerspointwiseestimations(model::DependentScalingModel, Tvalues::Vector{<:Real}, durations::Vector{<:Real})\n\nReturns the layers associated to pointwise estimations of the return levels at the differents durations, for every return period in T_values.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.plotIDFCurves-Tuple{DependentScalingModel, IDFdata}","page":"Types and Functions","title":"IDFCurves.plotIDFCurves","text":"plotIDFCurves(model::DependentScalingModel, data::IDFdata; ...)\n\nPlots the IDF curves based on the given model. Pointwise estimations are added (crosses) to illustrate the fitting. Durations and return periods may be chosen by the user but have default values. If showconfidenceintervals = true, 95% confidence intervals associated to the pointwise estimations are represented.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.plotIDFCurves-Tuple{DependentScalingModel}","page":"Types and Functions","title":"IDFCurves.plotIDFCurves","text":"plotIDFCurves(model::DependentScalingModel; ...)\n\nPots the IDF curves based on the given model. Durations and return periods may be chosen by the user but have default values.\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.qqplot-Tuple{Distributions.Distribution, Vector{Float64}}","page":"Types and Functions","title":"IDFCurves.qqplot","text":"qqplot(pd::Distribution, y::Vector{Float64})\n\nQuantile-Quantile plot from ExtendedExtremes.jl\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.qqplot-Tuple{MarginalScalingModel, IDFdata, Real}","page":"Types and Functions","title":"IDFCurves.qqplot","text":"qqplot(pd::Distribution, y::Vector{Float64})\n\nQuantile-Quantile plot from ExtendedExtremes.jl\n\n\n\n\n\n","category":"method"},{"location":"functions/#IDFCurves.qqplotci","page":"Types and Functions","title":"IDFCurves.qqplotci","text":"qqplotci(fm::DependentScalingModel, α::Real=.05, H::PDMat{<:Real})\n\nQuantile-Quantile plot along with the confidence/credible interval of level 1-α.\n\nDetails\n\nThis function uses the Hessian matrix H provided in the argument.  \n\n\n\n\n\n","category":"function"},{"location":"functions/#IDFCurves.qqplotci-2","page":"Types and Functions","title":"IDFCurves.qqplotci","text":"qqplotci(fm::MarginalScalingModel, α::Real=.05)\n\nQuantile-Quantile plot along with the confidence/credible interval of level 1-α.\n\n\n\n\n\n","category":"function"},{"location":"functions/#IDFCurves.qqplotci-3","page":"Types and Functions","title":"IDFCurves.qqplotci","text":"qqplotci(fm::MarginalScalingModel, α::Real=.05, H::PDMat{<:Real})\n\nQuantile-Quantile plot along with the confidence/credible interval of level 1-α.\n\nDetails\n\nThis function uses the Hessian matrix H provided in the argument.  \n\n\n\n\n\n","category":"function"},{"location":"functions/#IDFCurves.qqplotci-4","page":"Types and Functions","title":"IDFCurves.qqplotci","text":"qqplotci(fm::DependentScalingModel, data::IDFdata, d::Real, α::Real=.05)\n\nQuantile-Quantile plot for the estimated distribution for duration d, along with the confidence interval of level 1-α.\n\n\n\n\n\n","category":"function"},{"location":"tutorial/idf_estimation/#IDF-estimation-and-analysis-for-precipitation-recorded-at-the-Montréal-Pierre-Elliott-Trudeau-international-airport","page":"IDF estimation","title":"IDF estimation and analysis for precipitation recorded at the Montréal Pierre-Elliott-Trudeau international airport","text":"","category":"section"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"This tutorial illustrates the functionalities of the library. Before being able to execute it, the following libraries must be installed and imported.","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"using Cairo, CSV, DataFrames, Distributions, Fontconfig, Gadfly, LinearAlgebra, IDFCurves","category":"page"},{"location":"tutorial/idf_estimation/#Data-loading","page":"IDF estimation","title":"Data loading","text":"","category":"section"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Loading the IDF data recorded at Montréal Pierre-Elliott-Trudeau international airport:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"df = IDFCurves.dataset(\"702S006\")\nfirst(df,5)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Converting the DataFrame in a IDFdata structure. A dictionary mapping the tags (String) and the durations (Real) must be first defined:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"tags = names(df)[2:10]\ndurations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]\nduration_dict = Dict(zip(tags, durations))","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"The DataFrame can then be converted in a IDFdata structure:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"data = IDFdata(df, \"Year\", duration_dict)","category":"page"},{"location":"tutorial/idf_estimation/#Testing-which-scaling-model-may-be-suited-for-the-data","page":"IDF estimation","title":"Testing which scaling model may be suited for the data","text":"","category":"section"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Testing the simple scaling model:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"IDFCurves.scalingtest(SimpleScaling, data)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Testing the general scaling model:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"IDFCurves.scalingtest(GeneralScaling, data)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"In the first case, the very small p-value indicates that the simple scaling model is to be rejected. In the second case, the p-value is bigger than 005 (the value 10 is not relevant as the approximation is valid only for small p-values). Hence the general scaling model cannot be rejected.","category":"page"},{"location":"tutorial/idf_estimation/#Estimating-the-general-scaling-model-(also-known-as-d-GEV)","page":"IDF estimation","title":"Estimating the general scaling model (also known as d-GEV)","text":"","category":"section"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"The general scaling model is fitted to the data. 1 is the duration that should be used as a reference for parametrization. 20 5 04 76 07 is the vector of parameters used to initialize the optimization algorithm.","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"fmodel = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .04, .76, .07])","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"The initial vector of parameters is an optional argument as automatic initialization is available: ","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"fmodel = IDFCurves.fit_mle(GeneralScaling, data, 1)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Plotting the associated IDF curve:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Gadfly.set_default_plot_size(15cm, 8cm) #hide\nIDFCurves.plotIDFCurves(fmodel)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"The return levels estimated marginally for each duration may be added to the plot for illustration purposes:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Gadfly.set_default_plot_size(15cm, 8cm) #hide\nIDFCurves.plotIDFCurves(fmodel, data)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Displaying the model fit (using a qq-plot with confidence intervals) for the 5min, 1h, and 24h durations:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"p5min = qqplotci(fmodel, data, 5/60)\np1h = qqplotci(fmodel, data, 1)\np24h = qqplotci(fmodel, data, 24)\nGadfly.set_default_plot_size(30cm, 8cm) #hide\nhstack([p5min, p1h, p24h])","category":"page"},{"location":"tutorial/idf_estimation/#Estimating-the-general-scaling-model-in-combination-with-a-Gaussian-copula-and-the-Matern-correlation-structure","page":"IDF estimation","title":"Estimating the general scaling model in combination with a Gaussian copula and the Matern correlation structure","text":"","category":"section"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"fmodel2 = IDFCurves.fit_mle(DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}, data, 1.)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Plotting the associated IDF curve:c","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Gadfly.set_default_plot_size(15cm, 8cm) #hide\nIDFCurves.plotIDFCurves(fmodel2, data)","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"Displaying the model fit (using a qq-plot with confidence intervals) for the 5min, 1h, and 24h durations:","category":"page"},{"location":"tutorial/idf_estimation/","page":"IDF estimation","title":"IDF estimation","text":"p5min = qqplotci(fmodel2, data, 5/60)\np1h = qqplotci(fmodel2, data, 1)\np24h = qqplotci(fmodel2, data, 24)\nGadfly.set_default_plot_size(30cm, 8cm) #hide\nhstack([p5min, p1h, p24h])","category":"page"},{"location":"#Intensity-Duration-Curves-(IDF)-estimation-and-analysis-in-Julia","page":"Intensity Duration Curves (IDF) estimation and analysis in Julia","title":"Intensity Duration Curves (IDF) estimation and analysis in Julia","text":"","category":"section"},{"location":"","page":"Intensity Duration Curves (IDF) estimation and analysis in Julia","title":"Intensity Duration Curves (IDF) estimation and analysis in Julia","text":"IDFCurves.jl provides exhaustive high-performance functions for the estimation and the analysis of IDF curves in Julia. In particular, several models are provided, such as:","category":"page"},{"location":"","page":"Intensity Duration Curves (IDF) estimation and analysis in Julia","title":"Intensity Duration Curves (IDF) estimation and analysis in Julia","text":"independant GEV;\nsimple scaling;\ngeneral scaling;\ncomposite scaling;\nbreak scaling.","category":"page"}]
}
