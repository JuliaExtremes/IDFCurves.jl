
# IDF estimation and analysis for precipitation recorded at the Montréal Pierre-Elliott-Trudeau international airport

This tutorial illustrates the functionalities of the library. Before being able to execute it, the following libraries must be installed and imported.

```@example montreal
using Cairo, CSV, DataFrames, Distributions, Fontconfig, Gadfly, LinearAlgebra, IDFCurves
```

## Data loading

Loading the IDF data recorded at Montréal Pierre-Elliott-Trudeau international airport:

```@example montreal
df = IDFCurves.dataset("702S006")
first(df,5)
```

Converting the DataFrame in a [`IDFdata`](@ref) structure. A dictionary mapping the tags (String) and the durations (Real) must be first defined:

```@repl montreal
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
```

The DataFrame can then be converted in a [`IDFdata`](@ref) structure:

```@repl montreal
data = IDFdata(df, "Year", duration_dict)
```

## Testing which scaling model may be suited for the data

Testing the simple scaling model:

```@repl montreal
IDFCurves.scalingtest(SimpleScaling, data)
```

Testing the general scaling model:

```@repl montreal
IDFCurves.scalingtest(GeneralScaling, data)
```

In the first case, the very small p-value indicates that the simple scaling model is to be rejected. In the second case, the p-value is bigger than $0.05$ (the value $1.0$ is not relevant as the approximation is valid only for small p-values). Hence the general scaling model cannot be rejected.

## Estimating the general scaling model (also known as d-GEV)

The general scaling model is fitted to the data. $1$ is the duration that should be used as a reference for parametrization. $[20, 5, .04, .76, .07]$ is the vector of parameters used to initialize the optimization algorithm.

```@repl montreal
fmodel = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .04, .76, .07])
```

The initial vector of parameters is an optional argument as automatic initialization is available: 

```@repl montreal
fmodel = IDFCurves.fit_mle(GeneralScaling, data, 1)
```

Plotting the associated IDF curve:

```@repl montreal
Gadfly.set_default_plot_size(15cm, 8cm)
IDFCurves.plotIDFCurves(fmodel)
```

The return levels estimated marginally for each duration may be added to the plot for illustration purposes:

```@repl montreal
Gadfly.set_default_plot_size(15cm, 8cm)
IDFCurves.plotIDFCurves(fmodel, data)
```

Displaying the model fit (using a qq-plot with confidence intervals) for the 5min, 1h, and 24h durations:

```@repl montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p5min = qqplotci(fmodel, data, 5/60)
p1h = qqplotci(fmodel, data, 1)
p24h = qqplotci(fmodel, data, 24)
Gadfly.set_default_plot_size(30cm, 8cm)
hstack([p5min, p1h, p24h])
```

## Estimating the general scaling model in combination with a Gaussian copula and the Matern correlation structure

```@repl montreal
fmodel2 = IDFCurves.fit_mle(DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}, data, 1)
```

Plotting the associated IDF curve:c

```@repl montreal
Gadfly.set_default_plot_size(15cm, 8cm)
IDFCurves.plotIDFCurves(fmodel2, data)
```

Displaying the model fit (using a qq-plot with confidence intervals) for the 5min, 1h, and 24h durations:

```@repl montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p5min = qqplotci(fmodel2, data, 5/60)
p1h = qqplotci(fmodel2, data, 1)
p24h = qqplotci(fmodel2, data, 24)
Gadfly.set_default_plot_size(30cm, 8cm)
hstack([p5min, p1h, p24h])
```