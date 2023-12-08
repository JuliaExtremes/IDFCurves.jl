
@testset "matern" begin
    @test_throws AssertionError IDFCurves.matern(-1,1,1)
    @test_throws AssertionError IDFCurves.matern(1,-1,1)
    @test_throws AssertionError IDFCurves.matern(1,1,-1)

    @test IDFCurves.matern(0,1,1) ≈ 1
    @test IDFCurves.matern(2,1/2,1) ≈ exp(-2/1)
    @test IDFCurves.matern(2,3/2,1) ≈ (1+sqrt(3)*2)*exp(-sqrt(3)*2)
end

@testset "logdist(x₁::Real, x₂::Real)" begin
    @test_throws AssertionError IDFCurves.logdist(-1,1)
    @test_throws AssertionError IDFCurves.logdist(1,-1)

    @test IDFCurves.logdist(1,1) ≈ 0
    @test IDFCurves.logdist(exp(1),exp(2)) ≈ 1
end

@testset "logdist(x::AbstractVector{<:Real})" begin
    x = exp.(1:2)
    @test IDFCurves.logdist(x) ≈ [0. 1.; 1. 0.]
end