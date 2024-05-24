@testset "cvmcriterion()" begin
    distrib = Normal(0,1)
    x = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]

    @test IDFCurves.cvmcriterion(distrib,x) ≈ 0.31140110078245337 # obtained with scipy.stats.cramervonmises
end

fd = IDFCurves.SimpleScaling(1, 10, 2, 0.1, 0.7)

@testset "get_g()" begin
    
    g = IDFCurves.get_g(fd, 1)

    @test_throws AssertionError g(0)

    f(θ::AbstractArray) = [exp(θ[1]), exp(θ[2]), IDFCurves.logistic(θ[3])-.5, IDFCurves.logistic(θ[4])]
    jac = ForwardDiff.jacobian(f, [log(IDFCurves.location(fd)), log(IDFCurves.scale(fd)), IDFCurves.logit(IDFCurves.shape(fd)+0.5), IDFCurves.logit(IDFCurves.exponent(fd))])
    @test all( jac*g(0.8) ≈ [-0.7682503485212765, -0.2486477134711276, -0.04587320364441143, 0.0]) # explicit calculation with IDF.jl

    g = IDFCurves.get_g(fd, 2)
    @test !(g(0.5)[1] ≈ 0.)

end

@testset "approx_eigenvalues()" begin
    
    ρ(u,v) = u*v

    λs = IDFCurves.approx_eigenvalues(ρ, 10)
    @test length(λs) == 10
    @test λs[6] >= λs[7]

    λs = IDFCurves.approx_eigenvalues(ρ, 100)
    @test length(λs) == 100
    @test isapprox(λs[1], 1/3, rtol=1e-3)
    @test abs(λs[2]) <= 0.001

end

