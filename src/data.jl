"""
    dataset(name::String)::DataFrame

Load the dataset associated with `name`.

### Details

Some datasets are available using the following names:
 - `1108487`: short duration rainfall intensity-duration-frequency data recorded at VANCOUVER UBC
 - `6158731`: short duration rainfall intensity-duration-frequency data recorded at the Toronto Leaster B. Pearson Internation Airport
 - `702S006`: short duration rainfall intensity-duration-frequency data recorded at the Montreal Pierre-Elliott-Trudeau Internation Airport
 
 These datasets have been retrieved from the [Environment and Climate Change Canada website](https://climate.weather.gc.ca/prods_servs/engineering_e.html).

### Examples
```julia-repl
julia> IDFCurves.dataset("702S006")
```
"""
function dataset(name::String)::DataFrame

    filename = joinpath(dirname(@__FILE__), "..", "data", string(name, ".csv"))
    if isfile(filename)
        return CSV.read(filename, DataFrame)
    end
    error("There is no dataset with the name '$name'")

end