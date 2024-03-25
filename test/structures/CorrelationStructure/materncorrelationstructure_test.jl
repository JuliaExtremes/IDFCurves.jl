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

    @testset "construct_model(::Type{<:MaternCorrelationStructure}, θ)" begin

        θ = [0.]
        @test_throws AssertionError IDFCurves.construct_model(MaternCorrelationStructure, θ)
        
        θ = [0.,0.]
        cor_struct = IDFCurves.construct_model(MaternCorrelationStructure, θ)
        @test cor_struct isa MaternCorrelationStructure
        @test all([params(cor_struct)...] .≈  [1.,1.])
        
    end

    @testset "map_to_real_space(::Type{<:MaternCorrelationStructure}, θ)" begin
        
        θ = [1., 2.]
        @test IDFCurves.map_to_real_space(MaternCorrelationStructure, θ) ≈ [0., log(2)]
    end

end