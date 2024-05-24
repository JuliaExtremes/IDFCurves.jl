using IDFCurves
using CSV, DataFrames, Distributions, PDMats, SpecialFunctions, ForwardDiff, Test

@testset "IDFCurves.jl" begin
    include("utils_test.jl")
    include("data_test.jl")
    include("structures_test.jl")
    include("scalingtest_test.jl")
end
