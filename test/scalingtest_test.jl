@testset "cvmcriterion()" begin
    distrib = Normal(0,1)
    x = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]

    @test IDFCurves.cvmcriterion(distrib,x) ≈ 0.31140110078245337 # obtained with scipy.stats.cramervonmises
end

@testset "get_g()" begin

    fd = IDFCurves.SimpleScaling(1, 10, 2, 0.1, 0.7)

    g = IDFCurves.get_g(fd, 1)

    @test_throws AssertionError g(0)

    f1(θ::AbstractArray) = [exp(θ[1]), exp(θ[2]), IDFCurves.logistic(θ[3])-.5, IDFCurves.logistic(θ[4])]
    jac = ForwardDiff.jacobian(f1, [log(IDFCurves.location(fd)), log(IDFCurves.scale(fd)), IDFCurves.logit(IDFCurves.shape(fd)+0.5), IDFCurves.logit(IDFCurves.exponent(fd))])
    @test all( jac*g(0.8) ≈ [-0.7682503485212765, -0.2486477134711276, -0.04587320364441143, 0.0]) # explicit calculation with IDF.jl

    g = IDFCurves.get_g(fd, 2)
    @test !(g(0.5)[4] ≈ 0.)

    fd = GeneralScaling(1, 10, 2, 0.1, 0.7, 3/60)

    g = IDFCurves.get_g(fd, 1)
    f2(θ::AbstractArray) = [exp(θ[1]), exp(θ[2]), IDFCurves.logistic(θ[3])-.5, IDFCurves.logistic(θ[4]), exp(θ[5])/60]
    jac = ForwardDiff.jacobian(f2, [log(IDFCurves.location(fd)), log(IDFCurves.scale(fd)), IDFCurves.logit(IDFCurves.shape(fd)+0.5), IDFCurves.logit(IDFCurves.exponent(fd)), log(IDFCurves.offset(fd) * 60)])
    @test all( jac*g(0.7) ≈ [-1.1260765281539733, -0.24457155126318278, -0.030776128447083745, 0.0, 0.0]) # explicit calculation with IDF.jl

    g = IDFCurves.get_g(fd, 2)
    @test !(g(0.5)[5] ≈ 0.)

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

@testset "zolotarev_approx()" begin

    λs = Float64[]
    @test_throws AssertionError IDFCurves.zolotarev_approx(λs , 1.)

    λs = [1]
    @test isapprox( IDFCurves.zolotarev_approx(λs , Distributions.quantile(Distributions.Chisq(1), 0.99)), 0.99, rtol = 1e-2 )

    λs = fill(1,10)
    @test_logs (:warn,"Zolotarev approximation is outside its validity domain. No conclusion can be made from a small p-value.") IDFCurves.zolotarev_approx(λs , 0.)
    @test isapprox( IDFCurves.zolotarev_approx(λs , Distributions.quantile(Distributions.Chisq(10), 0.99)), 0.99, rtol = 1e-2 )

    λs = Float64[]
    for k in 1:10
        append!(λs, fill(1/2^k, 2*k^2))
    end
    @test isapprox( IDFCurves.zolotarev_approx(λs , 16.51), 0.99, rtol = 1e-2 ) # value 16.51 was obtained by simulation.

end

df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

@testset "scalingtest()" begin

    @test_throws ErrorException scalingtest(SimpleScaling, data, d_out = 1/60)
    @test scalingtest(SimpleScaling, data) ≈ 3.413181991929193e-6
    @test scalingtest(GeneralScaling, data, q = 50) >= 0.01

end

