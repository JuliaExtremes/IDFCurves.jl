# IDFCurves

[![Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/JuliaExtremes/IDFCurves.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaExtremes/IDFCurves.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JuliaExtremes/IDFCurves.jl/branch/main/graph/badge.svg?token=5fe36122-1af1-4494-be65-e307d5aa8acc)](https://codecov.io/gh/JuliaExtremes/IDFCurves.jl)
[![documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaExtremes.github.io/IDFCurves.jl/stable/)
[![documentation latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaExtremes.github.io/IDFCurves.jl/dev/)


IDFCurves.jl is a package in the **Julia** programming language ecosystem. It specializes in the statistical estimation of Intensity-Duration-Frequency curves. It provides exhaustive, high-performance functions by leveraging the multiple-dispatch capabilities in **Julia**. In particular, the package implements:

- Estimating several scaling models for IDF curves
- Performing goodness-of-fit tests for scaling models
- Modelling the dependence of maxima across durations
- Estimating the uncertainty of IDF curves
- Displaying IDF curve estimates and their uncertainty

IDFCurves.jl leverage on [Extremes.jl](https://github.com/jojal5/Extremes.jl) for the analysis of extreme values. 

## Documentation

See the [Package Documentation](https://JuliaExtremes.github.io/IDFCurves.jl/stable/) for details and examples on how to use the package and also the following publications:


References: 

Jalbert, J., Farmer, M., Gobeil, G., & Roy, P. (2024). Extremes.jl: Extreme Value Analysis in Julia. Journal of Statistical Software, 109(6), 1–35. https://doi.org/10.18637/jss.v109.i06

Mathivon, P., Genest, C., & Jalbert, J. (2024+). Joint modeling of annual precipitation maxima over several durations for the construction of intensity-duration-frequency curves. Preprint available soon.

Paoli, A., Carreau, J, & Jalbert, J. (2024+). Statistical testing of scaling models for precipitation Intensity-Duration-Frequency curves. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4838410


## Installation

The following **julia** command will install the package:

```julia
julia> Pkg.add("IDFCurves")
```

## Data
The datasets that are available through this package are retriveved from the [Environment and Climate Change Canada](https://collaboration.cmc.ec.gc.ca/cmc/climate/Engineer_Climate/IDF/) website.