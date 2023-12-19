using IDFCurves
using CSV, DataFrames, Distributions, PDMats, SpecialFunctions, Test

@testset "IDFCurves.jl" begin
    include("utils_test.jl")
    include("data_test.jl")
    include("structures_test.jl")
end
