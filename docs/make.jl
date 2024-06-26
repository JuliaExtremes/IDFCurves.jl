using Documenter, Cairo, Fontconfig
using IDFCurves

CI = get(ENV, "CI", nothing) == "true"

makedocs(sitename = "IDFCurves.jl",
    format = Documenter.HTML(
        prettyurls = CI, size_threshold_warn=10^8 ,size_threshold=10^9, example_size_threshold=10^9),
    pages = [
        "Tutorial" =>["IDF estimation" => "tutorial/idf_estimation.md"],
        "index.md",
    #    "contributing.md",
        "functions.md"
       ]
)

if CI
    deploydocs(
    repo   = "github.com/JuliaExtremes/QuantileMatching.jl.git",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#", "main", "dev"],
    push_preview = false,
    target = "build"
    )
end

# deploydocs(repo = "github.com/jojal5/Extremes.jl.git")