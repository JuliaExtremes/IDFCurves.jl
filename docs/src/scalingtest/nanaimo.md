# Nanaimo (BC) precipitation IDF data

```@setup nanaimo
using Cairo, CSV, DataFrames, Distributions, Fontconfig, Gadfly, IDFCurves, LinearAlgebra, Random
```

```@setup nanaimo
Gadfly.set_default_plot_size(12cm, 8cm)
```

Load the IDF data recorded at Nanaimo (BC) Airport:

```@example nanaimo
df = IDFCurves.dataset("1025369")
first(df, 5)
```

Convert the resulting `DataFrame` into an [`IDFdata`](@ref) structure. First,
define a dictionary mapping each duration tag to its duration in hours:

```@example nanaimo
tags = names(df)[2:10]
durations = [1 / 12, 1 / 6, 1 / 4, 1 / 2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
nothing #hide
```

The `DataFrame` can then be converted into an [`IDFdata`](@ref) structure:

```@example nanaimo
data = IDFdata(df, "Year", duration_dict)
```

## Testing the simple scaling assumption

The simple scaling model is fitted to all durations except the 5-minute
duration. The fitted model is then used to test the implied distribution of the
held-out 5-minute data.

Run the training-validation Cramér--von Mises test and compute its analytical
p-value:

```@example nanaimo
T = scalingtest(SimpleScaling, data; tag_out="5min")

analytical_pvalue = IDFCurves.pvalue(T)
```

The analytical p-value is large, not providing strong evidence against the
simple scaling assumption.

### Bootstrap calibration

Annual precipitation maxima at different durations within the same year may be
dependent. This inter-duration dependence can be accounted for using a
year-wise bootstrap. The following example uses 999 bootstrap replicates:

```@example nanaimo
B = 999
rng = Xoshiro(1234)

Sstar = IDFCurves.scalingtest_bootstrap(T.fitted_model, data; tag_out="5min", B=B, rng=rng)

S = T.test_statistic

bootstrap_pvalue =
    (1 + count(s -> s >= S, Sstar)) / (B + 1)
```

The bootstrap p-value is also large, not providing sufficient evidence
against the simple scaling assumption.

### Visual comparison

As a visual diagnostic, compare the distribution predicted by the simple
scaling model fitted without the 5-minute duration with the empirical
distribution of the held-out 5-minute observations.

The fitted model returned by the test can be reused directly:

```@example nanaimo
ss_model = T.fitted_model

distribution_5min =
    IDFCurves.getdistribution(ss_model, 5 / 60)
```

Construct the empirical distribution function of the 5-minute observations:

```@example nanaimo
y_5min = sort(IDFCurves.getdata(data, "5min"))

empirical_cdf(x) =
    searchsortedlast(y_5min, x) / (length(y_5min) + 1)

nothing #hide
```

Compare the predicted and empirical distributions:

```@example nanaimo
fig = plot([
        x -> cdf(distribution_5min, x),
        x -> empirical_cdf(x),],
    0, 250,
    Guide.xlabel("5-minute precipitation intensity (mm/h)"),
    Guide.ylabel("Probability"),
    Theme(key_position=:none),
)

fig #hide
# draw(PDF("Nan_cdf_simplescaling.pdf"), fig) #hide
```
