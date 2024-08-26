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

# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# data that raises an error during hessian computation
df = DataFrame(Year = 1:5,
                d1 = [212.88548497971587, 186.21870723630911, 192.5941589629015, 207.70771398543727, 253.28098069877467],
                d2 = [115.66936090096817, 102.07530027790483, 110.23552647428488, 101.43374631705547, 114.72830627764733],
                d3 = [85.24581170746404, 65.12966286341307, 76.7475813220242, 75.79353124455645, 68.56626670369397],
                d4 = [34.319734364436265, 31.72203736891103, 41.73319790518032, 42.38698775756336, 48.960074714341765],
                d5 = [22.52177638650152, 17.26322717397859, 19.168603864047387, 23.452621217405845, 20.953617397406624],
                d6 = [11.824403876924093, 9.714806478882196, 12.317907264280239, 11.186164746121602, 8.668935072544908],
                d7 = [6.150480234167272, 6.864025692753223, 5.6060380841178965, 5.4371321871641705, 8.222157251560155],
                d8 = [2.6727983163545854, 3.824937599625118, 3.5706517287382065, 3.6231397223924047, 3.2366781273031098],
                d9 = [1.8296778639474427, 1.7248572090544316, 2.2044817450846086, 2.061289389224367, 2.078686647048528])
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data_bug = IDFdata(df, "Year", duration_dict)

@testset "scalingtest()" begin

    @test_throws AssertionError scalingtest(SimpleScaling, data, "1min")
    @test scalingtest(SimpleScaling, data) ≈ 3.413181991929193e-6
    @test scalingtest(GeneralScaling, data, "5min", 50) >= 0.01

    # @test scalingtest(GeneralScaling, data_bug, "d1") ≈ 1.

end