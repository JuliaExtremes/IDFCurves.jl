
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

The DataFrame can then be converted in a [`IDFdata`](@ref) structure
```@repl montreal
data = IDFdata(df, "Year", duration_dict)
```

## Estimating the general scaling model (also known as d-GEV)

```@repl montreal
fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [20, 5, .04, .76, .07])
```

The hessian matrix evaluated at the point estimates can be approximated as follows: 
```@repl montreal
H = IDFCurves.hessian(fd, data)
```

The Wald point estimates distributions can be defined as follows:
```@repl montreal
Σ = inv(H)

v = diag(Σ)

W = Normal.(collect(params(fd)), sqrt.(v))
```

The corresponding approximate confidence intervals can be obtained as follows:
```@repl montreal
bounds = quantile.(W, [.025 .975])
```


Displaying the model fit for the 5-, 10-, and 15-minute durations:
```@example montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p5min = qqplotci(fd, data, 5/60)
p10min = qqplotci(fd, data, 10/60)
p15min = qqplotci(fd, data, 15/60)

hstack([p5min, p10min, p15min])
```

Displaying the model fit for the 30-min-, 1-, and 2-hour durations:
```@example montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p30min = qqplotci(fd, data, 30/60)
p1h = qqplotci(fd, data, 1)
p2h = qqplotci(fd, data, 2)

hstack([p30min, p1h, p2h])
```

Displaying the model fit for the 30-min-, 1-, and 2-hour durations:
```@example montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p6h = qqplotci(fd, data, 6)
p12h = qqplotci(fd, data, 12)
p24h = qqplotci(fd, data, 24)

hstack([p6h, p12h, p24h])
```



## Estimating the general scaling model (also known as d-GEV) in combination with a Gaussian copula


```@repl montreal
initialvalues = [20, 5, .04, .76, .07, 3]
fm = IDFCurves.fit_mle(DependentScalingModel{dGEV, GaussianCopula}, data, 1, initialvalues)
```


Displaying the model fit for the 5-, 10-, and 15-minute durations:
```@example montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p5min = qqplotci(fm, data, 5/60)
p10min = qqplotci(fm, data, 10/60)
p15min = qqplotci(fm, data, 15/60)
    
hstack([p5min, p10min, p15min])
```
    
Displaying the model fit for the 30-min-, 1-, and 2-hour durations:
```@example montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p30min = qqplotci(fm, data, 30/60)
p1h = qqplotci(fm, data, 1)
p2h = qqplotci(fm, data, 2)
    
hstack([p30min, p1h, p2h])
```
    
Displaying the model fit for the 30-min-, 1-, and 2-hour durations:
```@example montreal
Gadfly.set_default_plot_size(30cm, 8cm)
p6h = qqplotci(fm, data, 6)
p12h = qqplotci(fm, data, 12)
p24h = qqplotci(fm, data, 24)
    
hstack([p6h, p12h, p24h])
```
    



