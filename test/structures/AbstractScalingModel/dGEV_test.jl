
@testset "dGEV construction" begin
    pd = dGEV(60, 100, 1, 0, .8, 5)
    @test location(pd) == 100
    @test scale(pd) == 1
    @test shape(pd) == 0
    @test exponent(pd) ≈ .8 
    @test offset(pd) == 5
    @test duration(pd) == 60
    @test all([params(pd)...] .≈ [100, 1, 0, .8, 5])
end

@testset "getdistribution(::dGEV)" begin
    pd = dGEV(60, 100, 1, 0, .8, 5)
    
    md = getdistribution(pd, 3*60)
    
    @test location(md) ≈ 43.31051048132165
    @test scale(md) ≈ 0.4331051048132165
    @test shape(md) ≈ 0.
    
end

@testset "map_to_param_space(::Type{<:dGEV}, θ)" begin
    
    θ = [1., 0., 0., 0., 0.]
    @test IDFCurves.map_to_param_space(dGEV, θ) ≈ [1., 1., 0., .5, 1.]
end

@testset "Base.show(io, dGEV)" begin
    # print dGEV does not throw
    pd = dGEV(60, 100, 1, 0, .8, 5)
    buffer = IOBuffer()
    @test_logs Base.show(buffer, pd)

end