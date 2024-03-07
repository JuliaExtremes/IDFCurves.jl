@testset "ExponentialCorrelationStructure" begin
    
    @testset "ExponentialCorrelationStructure contructor" begin
        @test params(ExponentialCorrelationStructure(1)) ≈ (1.0)
        @test_throws AssertionError ExponentialCorrelationStructure(-1)
    end
    
    @testset "cor(::ExponentialCorrelationStructure)" begin
        C = ExponentialCorrelationStructure(1)    
    
        @test_throws AssertionError cor(C,-1)
        @test cor(C,1) ≈ exp(-1)
        @test cor.(C,[1, 2]) ≈ exp.(-[1, 2])
    end

    @testset "map_to_param_space(::Type{<:ExponentialCorrelationStructure}, θ)" begin
        
        θ = [0.]
        @test IDFCurves.map_to_param_space(ExponentialCorrelationStructure, θ) ≈ [1.]
    end

    @testset "map_to_real_space(::Type{<:ExponentialCorrelationStructure}, θ)" begin
        
        θ = [1.]
        @test IDFCurves.map_to_real_space(ExponentialCorrelationStructure, θ) ≈ [0.]
    end

end