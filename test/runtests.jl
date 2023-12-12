using IDFCurves
using CSV, DataFrames, Distributions, PDMats, Test

@testset "IDFCurves.jl" begin
    include("utils_test.jl")
    include("data_test.jl")
    include("structures_test.jl")
end
