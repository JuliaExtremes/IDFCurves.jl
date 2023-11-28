@testset "structures.jl" begin
    include(joinpath("structures", "idfdata_test.jl"))
    include(joinpath("structures","AbstractScalingModel", "abstractscalingmodel_test.jl"))
end