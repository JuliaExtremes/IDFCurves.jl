# Simulations for studying test characteristics

```@setup simulation
using CSV, DataFrames, IDFCurves
using Cairo, Fontconfig, Gadfly
using CategoricalArrays
```

This section presents precomputed simluation results. The results
can be reproduced using the `Simulations/simulations.jl` script available in this
repository.

This section requires the additional package *CategoricalArrays.jl*

```@example simulation
using CategoricalArrays
```

## Type I error

### Simple Scaling

Load the precomputed results:
```@example simulation
filename = normpath(joinpath(@__DIR__, "..","..","..", "Simulations","simulation_results", "SimpleScaling_type1_error.csv"))
df = CSV.read(filename, DataFrame)
nothing #hide
```

Display the type I error when the true model is the simple scaling
```@example simulation
df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:n, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"),
    Coord.Cartesian(ymin=.04, ymax=.06)
)
# filename = joinpath(@__FILE__, folderpath, "simplescaling_errorI.pdf") #hide
# draw(PDF(filename), fig) #hide
fig #hide
```

### General Scaling

Load the precomputed results:
```@example simulation
filename = normpath(joinpath(@__DIR__, "..","..","..", "Simulations","simulation_results", "GeneralScaling_type1_error.csv"))
df = CSV.read(filename, DataFrame)
nothing #hide
```

Display the type I error when the true model is the general scaling
```@example simulation
df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:n, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"))

# filename = joinpath(@__FILE__, folderpath, "generalscaling_errorI.pdf") #hide
# draw(PDF(filename), fig) #hide
fig #hide
```

## Test power

### Simple scaling

Load the precomputed results:
```@example simulation
filename = normpath(joinpath(@__DIR__, "..","..","..", "Simulations","simulation_results", "SimpleScaling_power.csv"))
df = CSV.read(filename, DataFrame)
nothing #hide
```
Display the test power when the true model is the General Scaling with parameter δ
```@example simulation
df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:δ, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"))

# filename = joinpath(@__FILE__, folderpath, "simplescaling_power.pdf") #hide
# draw(PDF(filename), fig) #hide
fig #hide
```

### General scaling

Load the precomputed results:
```@example simulation
filename = normpath(joinpath(@__DIR__, "..","..","..", "Simulations","simulation_results", "GeneralScaling_power.csv"))
df = CSV.read(filename, DataFrame)
nothing #hide
```
Display the test power when the true model is the Hybrid Scaling with parameter α₁
```@example simulation
df.ξ = categorical(string.(df.ξ))

fig = plot(df, x=:α₁, y=:RejectionRate, color=:ξ, Geom.line, Geom.point,
    Guide.ylabel("Rejection Rate"))

# filename = joinpath(@__FILE__, folderpath, "generalscaling_power.pdf")) #hide
# draw(PDF(filename), fig) #hide
fig #hide
```