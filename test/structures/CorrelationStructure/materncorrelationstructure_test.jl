@testset "MaternCorrelationStructure" begin
    
    @testset "MaternCorrelationStructure construction" begin
        @test_throws "AssertionError" MaternCorrelationStructure(-1., 2.)
        @test_throws "AssertionError" MaternCorrelationStructure(1., -2.)
        
        C = MaternCorrelationStructure(1, 2)

        @test all(params(C) .≈ (1., 2.))
        @test typeof(params(C)) == Tuple{Float64, Float64}
        @test params_number(MaternCorrelationStructure) == 2
        
    end

    @testset "cor(::MaternCorrelationStructure)" begin

        C = MaternCorrelationStructure(10., 1.)
        d  = 2.

        ν, ρ = params(C)
        z = sqrt(2*ν)*d/ρ

        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * z^ν * SpecialFunctions.besselk(ν, z)

        @test cor(C, d) ≈ c
        
    end

    @testset "map_to_param_space(::Type{<:MaternCorrelationStructure}, θ)" begin
        
        θ = [0., -1.]
        @test IDFCurves.map_to_param_space(MaternCorrelationStructure, θ) ≈ [1., exp(-1)]
    end

    @testset "map_to_real_space(::Type{<:MaternCorrelationStructure}, θ)" begin
        
        θ = [1., 2.]
        @test IDFCurves.map_to_real_space(MaternCorrelationStructure, θ) ≈ [0., log(2)]
    end

end