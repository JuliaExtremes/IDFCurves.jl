@testset "MaternCorrelationStructure" begin
    
    @testset "MaternCorrelationStructure construction" begin
        @test_throws "AssertionError" MaternCorrelationStructure(-1., 2.)
        @test_throws "AssertionError" MaternCorrelationStructure(1., -2.)
        
        C = MaternCorrelationStructure(1, 2)

        @test all(params(C) .≈ (1., 2.))

        @test typeof(params(C)) == Tuple{Float64, Float64}
        
    end

    @testset "cor(::MaternCorrelationStructure)" begin

        C = MaternCorrelationStructure(10., 1.)
        d  = 2.

        ν, ρ = params(C)
        z = sqrt(2*ν)*d/ρ

        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * z^ν * SpecialFunctions.besselk(ν, z)

        @test cor(C, d) ≈ c
        
    end

end