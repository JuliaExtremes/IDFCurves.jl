using IDFCurves
using CSV, DataFrames, Distributions, Test

@testset "IDFCurves.jl" begin
    include("covariance_test.jl")
    include("data_test.jl")
    include("structures_test.jl")
end
