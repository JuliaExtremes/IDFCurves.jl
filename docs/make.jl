using Documenter, IDFCurves

CI = get(ENV, "CI", nothing) == "true"

makedocs(sitename = "IDFCurves.jl",
    format = Documenter.HTML(
    prettyurls = CI,
    ),
    pages = [
        "Tutorial" =>["IDF estimation" => "tutorial/idf_estimation.md"],
        "index.md",
    #    "contributing.md",
        "functions.md"
       ]
)

if CI
    deploydocs(
    repo   = "https://github.com/JuliaExtremes/IDFCurves.jl.git",
    devbranch = "dev",
    versions = ["stable" => "v^", "v#.#", "master"],
    push_preview = false,
    target = "build"
    )
end

# deploydocs(repo = "github.com/jojal5/Extremes.jl.git")