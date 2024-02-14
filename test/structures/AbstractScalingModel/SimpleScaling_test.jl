
@testset "SimpleScaling construction" begin
    pd = SimpleScaling(2, 100, 1, 0.1, .8)
    @test location(pd) == 100
    @test scale(pd) == 1
    @test shape(pd) ≈ 0.1
    @test exponent(pd) ≈ .8 
    @test duration(pd) == 2
    @test all([params(pd)...] .≈ [100, 1, 0.1, .8])
end

@testset "getdistribution(::SimpleScaling)" begin
    pd = SimpleScaling(1, 100, 4, 0.1, .8)
    
    md = getdistribution(pd, 3*1)
    
    @test location(md) ≈ 41.52436465385057
    @test scale(md) ≈ 1.6609745861540228
    @test shape(md) ≈ 0.1
    
end

@testset "map_to_param_space(::Type{<:SimpleScaling}, θ)" begin
    
    θ = [1., 0., 0., 0.]
    @test IDFCurves.map_to_param_space(SimpleScaling, θ) ≈ [1., 1., 0., .5]
end

@testset "Base.show(io, SimpleScaling)" begin
    # print dGEV does not throw
    pd = SimpleScaling(1, 100, 1, 0, .8)
    buffer = IOBuffer()
    @test_logs Base.show(buffer, pd)

end