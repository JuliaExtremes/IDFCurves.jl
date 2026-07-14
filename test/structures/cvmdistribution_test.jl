@testset "CvMDistribution" begin

    import IDFCurves: CvMDistribution

    @testset "CvMDistribution with positive weights" begin
        λ = collect(0.15:-0.01:0.05)

        d = CvMDistribution(λ)

        @test params(d)[1] ≈ λ
        @test length(d) == length(λ)
        @test minimum(d) == 0.0
        @test maximum(d) == Inf

        @test insupport(d, 0.0)
        @test insupport(d, 1.0)
        @test !insupport(d, -eps())

        xs = [1.0, 2.0, 5.0]
        ps = ccdf.(d, xs)
        # Expected values obtained using ccdf on GeneralizedChisq(λ, ones(Float64, length(λ)), zeros(Float64, length(λ)), 0.0, 0.0) from GeneralizedChisqDistribution.jl
        ps = [0.5207596726624824
            0.05235770081094071
            4.116282534383231e-6]

        for i in eachindex(xs)
            @test ccdf(d, xs[i]) ≈ ps[i] rtol = 1e-7 atol = 1e-8
        end

        for i in eachindex(ps)
            @test quantile(d, 1. - ps[i]) ≈ xs[i] rtol = 1e-5 atol = 1e-8
        end
    end

    @testset "CvMDistribution constructor checks" begin
        @test_throws ArgumentError CvMDistribution([0.7, 0.0, 0.2])
        @test_throws ArgumentError CvMDistribution(Float64[])
        @test_throws ArgumentError CvMDistribution([0.7, -0.2])
        @test_throws ArgumentError CvMDistribution([0.7, Inf])
        @test_throws ArgumentError CvMDistribution([0.7, NaN])
    end

end