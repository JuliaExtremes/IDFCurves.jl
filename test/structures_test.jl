@testset "structures.jl" begin

    include(joinpath("structures", "idfdata_test.jl"))
    include(joinpath("structures","MarginalScalingModel", "abstractmarginalscalingmodel_test.jl"))
    include(joinpath("structures","DependentScalingModel", "dependentscalingmodel_test.jl"))
    include(joinpath("structures","EllipticalCopula", "abstractellipticalcopula_test.jl"))
    include(joinpath("structures", "CorrelationStructure", "abstractcorrelationstructure_test.jl"))
end