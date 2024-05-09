@testset "ExponentialCorrelationStructure" begin
    
    @testset "ExponentialCorrelationStructure contructor" begin
        @test params(ExponentialCorrelationStructure(1)) ≈ (1.0)
        @test_throws AssertionError ExponentialCorrelationStructure(-1)
        @test params_number(ExponentialCorrelationStructure) == 1
    end
    
    @testset "cor(::ExponentialCorrelationStructure)" begin
        C = ExponentialCorrelationStructure(1)    
    
        @test_throws AssertionError cor(C,-1)
        @test cor(C,1) ≈ exp(-1)
        @test cor.(C,[1, 2]) ≈ exp.(-[1, 2])
    end

    @testset "construct_model(::Type{<:ExponentialCorrelationStructure}, θ)" begin

        θ = [0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(ExponentialCorrelationStructure, θ)
        
        θ = [0.]
        cor_struct = IDFCurves.construct_model(ExponentialCorrelationStructure, θ)
        @test cor_struct isa ExponentialCorrelationStructure
        @test all([params(cor_struct)...] .≈  [1.])
        
    end

    @testset "map_to_real_space(::Type{<:ExponentialCorrelationStructure}, θ)" begin
        
        θ = [1.]
        @test IDFCurves.map_to_real_space(ExponentialCorrelationStructure, θ) ≈ [0.]
    end

    df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
    tags = names(df)[2:10]
    durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
    duration_dict = Dict(zip(tags, durations))
    data = IDFdata(df, "Year", duration_dict)

    @testset "initialize(::Type{<:ExponentialCorrelationStructure}, data)" begin
        
        init = initialize(ExponentialCorrelationStructure, data)
        @test length(init) == 1
        @test init[1] ≈ 2.707665345991317

    end

end