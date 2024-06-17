using IDFCurves
using CSV, DataFrames, Distributions, ForwardDiff, PDMats, Random, SpecialFunctions, Test

@testset "IDFCurves.jl" begin
    include("utils_test.jl")
    include("data_test.jl")
    include("structures_test.jl")
    include("scalingtest_test.jl")
    include("misspecification_test.jl")
end
