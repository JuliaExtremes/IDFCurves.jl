

@testset "CvMValidationTest" begin

    import IDFCurves: CvMValidationTest, pvalue, decision_threshold, decision

    fitted_model = (; name = "dummy model")
    test_struct = CvMValidationTest(fitted_model, 1.96, Normal(0,1))

    @test test_struct.fitted_model == fitted_model
    @test test_struct.test_statistic == 1.96
    @test test_struct.null_distribution == Normal()

    @test pvalue(test_struct) ≈ ccdf(Normal(), 1.96)
    @test pvalue(test_struct) ≈ 0.024997895148220435

    @test decision_threshold(test_struct, 0.05) ≈ quantile(Normal(), 0.95)
    @test decision_threshold(test_struct, 0.05) ≈ 1.6448536269514717

    @test decision(test_struct, 0.05)
    @test !decision(test_struct, 0.01)

    @test_throws ArgumentError decision_threshold(test_struct, 0.0)
    @test_throws ArgumentError decision_threshold(test_struct, 1.0)
    @test_throws ArgumentError decision(test_struct, 0.0)
    @test_throws ArgumentError decision(test_struct, 1.0)
end

@testset "CvMValidationTest show" begin
    test_struct = CvMValidationTest(:model, 1.96, Normal())

    str = sprint(show, test_struct)

    @test occursin("CvMValidationTest(", str)
    @test occursin("  test_statistic = 1.96", str)
    @test occursin("  null_distribution = Normal{Float64}", str)
    @test occursin("  fitted_model = Symbol", str)
    @test endswith(str, ")")
end



@testset "cvmcriterion()" begin
    distrib = Normal(0,1)
    x = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]

    @test IDFCurves.cvmcriterion(distrib,x) ≈ 0.31140110078245337 # obtained with scipy.stats.cramervonmises
end

@testset "get_g()" begin

    fd = IDFCurves.SimpleScaling(1, 10, 2, 0.1, 0.7)

    g = IDFCurves.get_g(fd, 1)

    @test_throws ArgumentError g(0)

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

    # Brownian bridge kernel:
    # ρ(u, v) = min(u, v) - u*v
    #
    # It is obtained from cvmkernel by setting the correction term to zero.
    g(u) = [0.0]
    A = Matrix{Float64}(I, 1, 1)

    ρ = IDFCurves.cvmkernel(g, A)

    λs = IDFCurves.approx_eigenvalues(ρ, 10)

    @test length(λs) == 10
    @test issorted(λs; rev = true)
    @test all(λs .>= -sqrt(eps(Float64)))

    λs = IDFCurves.approx_eigenvalues(ρ, 100)

    @test length(λs) == 100
    @test issorted(λs; rev = true)

    # Exact eigenvalues of the Brownian bridge covariance kernel
    @test isapprox(λs[1], 1 / π^2, rtol = 2e-3)
    @test isapprox(λs[2], 1 / (4π^2), rtol = 3e-3)
    @test isapprox(λs[3], 1 / (9π^2), rtol = 5e-3)
end


@testset "zolotarev_approx()" begin

    λs = [0.0, -1.0]
    @test_throws ArgumentError IDFCurves.zolotarev_approx(λs, 1.0)

    λs = [1.0]
    x = Distributions.quantile(Distributions.Chisq(1), 0.99)
    @test isapprox(
        IDFCurves.zolotarev_approx(λs, x),
        0.99;
        rtol = 1e-2,
    )

    λs = fill(1.0, 10)

    @test_throws ArgumentError IDFCurves.zolotarev_approx(λs, 0.0)

    x = Distributions.quantile(Distributions.Chisq(10), 0.50)
    @test_logs (:warn, r"outside its recommended upper-tail domain") begin
        IDFCurves.zolotarev_approx(λs, x)
    end

    x = Distributions.quantile(Distributions.Chisq(10), 0.99)
    @test isapprox(
        IDFCurves.zolotarev_approx(λs, x),
        0.99;
        rtol = 1e-2,
    )

    λs = Float64[]
    for k in 1:10
        append!(λs, fill(1 / 2^k, 2k^2))
    end

    # The value 16.51 was obtained by simulation.
    @test isapprox(
        IDFCurves.zolotarev_approx(λs, 16.51),
        0.99;
        rtol = 1e-2,
    )

end



# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# # data that raises an error during hessian computation
# df = DataFrame(Year = 1:5,
#                 d1 = [212.88548497971587, 186.21870723630911, 192.5941589629015, 207.70771398543727, 253.28098069877467],
#                 d2 = [115.66936090096817, 102.07530027790483, 110.23552647428488, 101.43374631705547, 114.72830627764733],
#                 d3 = [85.24581170746404, 65.12966286341307, 76.7475813220242, 75.79353124455645, 68.56626670369397],
#                 d4 = [34.319734364436265, 31.72203736891103, 41.73319790518032, 42.38698775756336, 48.960074714341765],
#                 d5 = [22.52177638650152, 17.26322717397859, 19.168603864047387, 23.452621217405845, 20.953617397406624],
#                 d6 = [11.824403876924093, 9.714806478882196, 12.317907264280239, 11.186164746121602, 8.668935072544908],
#                 d7 = [6.150480234167272, 6.864025692753223, 5.6060380841178965, 5.4371321871641705, 8.222157251560155],
#                 d8 = [2.6727983163545854, 3.824937599625118, 3.5706517287382065, 3.6231397223924047, 3.2366781273031098],
#                 d9 = [1.8296778639474427, 1.7248572090544316, 2.2044817450846086, 2.061289389224367, 2.078686647048528])
# tags = names(df)[2:10]
# durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
# duration_dict = Dict(zip(tags, durations))
# data_bug = IDFdata(df, "Year", duration_dict)

@testset "scalingtest()" begin

    @test_throws ArgumentError scalingtest(SimpleScaling, data, tag_out = "1min")

    test_struct = scalingtest(SimpleScaling, data, tag_out = "5min", q=20)

    @test test_struct.test_statistic ≈ 3.344007260766803 atol=1e-8
    
    # @test scalingtest(GeneralScaling, data_bug, "d1") ≈ 1.

end