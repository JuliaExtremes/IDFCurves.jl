
# IDF estimation and analysis for precipitation recorded at the Montréal Pierre-Elliott-Trudeau international airport

This tutorial illustrates the functionalities of the library. Before being able to execute it, the following libraries must be installed and imported.

```@example montreal
using CSV, DataFrames, Distributions, Gadfly, IDFCurves
```

## Data loading

Loading the IDF data recorded at Montréal Pierre-Elliott-Trudeau international airport:
```@example montreal
df = IDFCurves.dataset("702S006")
first(df,5)
```

Converting the DataFrame in a [`IDFdata`](@ref) structure. A dictionary mapping the tags (String) and the durations (Real) must be first defined:
```@example montreal
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
```

The DataFrame can then be converted in a [`IDFdata`](@ref) structure
```@example montreal
data = IDFdata(df, "Year", duration_dict)
```

## Estimating the general scaling model (also know as d-GEV)

```@example montreal
fd = IDFCurves.fit_mle(dGEV, data, 1, [1., 1., .1, .8, .01])
```
