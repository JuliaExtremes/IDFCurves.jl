
@testset "Godambe information matrix()" begin
    Random.seed!(12345)
    @testset "without misspecification" begin
        # True model: Exponential
        # Fitted model: Exponential
        θ = 1.
        n = 10000
        y = rand(Exponential(1/θ), n);

        θ̂ = 1/mean(y)
        fd = Exponential(1/θ̂)

        J = IDFCurves.variability_matrix(fd, y)
        @test J[1] ≈ n rtol=.05

        H = IDFCurves.hessian(fd, y)
        @test H[1] ≈ n rtol=.05

        G = IDFCurves.godambe(fd, y)
        @test G[1] ≈ n rtol=.05

    end
    @testset "with misspecification" begin
        # True model: Gamma
        # Fitted model: Exponential
        θ = 1.
        n = 10000
        y = rand(Gamma(.5, 1), 10000)

        θ̂ = 1/mean(y)
        fd = Exponential(1/θ̂)

        G = IDFCurves.godambe(fd, y)
        @test G[1] ≈ 2*n rtol=.05

    end
end