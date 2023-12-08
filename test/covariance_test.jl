
@testset "matern" begin
    @test_throws AssertionError IDFCurves.matern(-1,1,1)
    @test_throws AssertionError IDFCurves.matern(1,-1,1)
    @test_throws AssertionError IDFCurves.matern(1,1,-1)

    @test IDFCurves.matern(0,1,1) ≈ 1
    @test IDFCurves.matern(2,1/2,1) ≈ exp(-2/1)
    @test IDFCurves.matern(2,3/2,1) ≈ (1+sqrt(3)*2)*exp(-sqrt(3)*2)
end