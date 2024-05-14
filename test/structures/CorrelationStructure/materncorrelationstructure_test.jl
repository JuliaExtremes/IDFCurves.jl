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

    df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
    tags = names(df)[2:10]
    durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
    duration_dict = Dict(zip(tags, durations))
    data = IDFdata(df, "Year", duration_dict)

    @testset "initialize(::Type{<:MaternCorrelationStructure}, data)" begin
        
        init = initialize(MaternCorrelationStructure, data)
        @test length(init) == 2
        @test init[1] ≈ 1.0026461842401198
        @test init[2] ≈ 2.189742232486837

    end

end