

@testset "AbstractCorrelationStructure" begin

    @testset "ExponentialCorrelationStructure contructor" begin
        @test params(ExponentialCorrelationStructure(1)) ≈ [1.0]
        @test_throws AssertionError ExponentialCorrelationStructure(-1)
    end
    
    @testset "cor(::ExponentialCorrelationStructure)" begin
        C = ExponentialCorrelationStructure(1)    
    
        @test_throws AssertionError cor(C,-1)
        @test cor(C,1) ≈ exp(-1)
        @test cor.(C,[1, 2]) ≈ exp.(-[1, 2])
    end
    
end
