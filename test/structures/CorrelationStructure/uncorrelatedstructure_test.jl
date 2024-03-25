@testset "UncorrelatedStructure" begin
    
    @testset "UncorrelatedStructure contructor" begin
        @test params(UncorrelatedStructure()) == ()
        @test params_number(UncorrelatedStructure) == 0
    end
    
    @testset "cor(::UncorrelatedStructure)" begin
        C = UncorrelatedStructure()    
    
        @test_throws AssertionError cor(C,-1)
        @test cor(C,0) == 1.
        @test cor(C,.1) == 0.
    end

    @testset "construct_model(::Type{<:UncorrelatedStructure}, θ)" begin

        θ = [0.]
        @test_throws AssertionError IDFCurves.construct_model(UncorrelatedStructure, θ)
        
        θ = Float64[]
        cor_struct = IDFCurves.construct_model(UncorrelatedStructure, θ)
        @test cor_struct isa UncorrelatedStructure

    end

    @testset "map_to_real_space(::Type{<:UncorrelatedStructure}, θ)" begin
        
        θ = Float64[]
        @test IDFCurves.map_to_real_space(UncorrelatedStructure, θ) == Float64[]

    end

end