
@testset "Godambe information matrix()" begin
    @testset "without misspecification" begin
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
end