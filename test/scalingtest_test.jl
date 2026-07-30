@testset "scalingtest_test.jl" begin

    @testset "CvMValidationTest" begin

        import IDFCurves: CvMValidationTest, pvalue, decision_threshold, decision

        fitted_model = (; name="dummy model")
        test_struct = CvMValidationTest(fitted_model, 1.96, Normal(0, 1))

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

        invalid_test = CvMValidationTest(fitted_model, 0.25, nothing)
        @test !IDFCurves.isvalid(invalid_test)
        @test_throws ArgumentError IDFCurves._null_distribution(invalid_test)
        @test_throws ArgumentError pvalue(invalid_test)
        @test_throws ArgumentError decision_threshold(invalid_test, .05)
        @test_throws ArgumentError decision(invalid_test, .05)
        
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
        distrib = Normal(0, 1)
        x = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]

        @test IDFCurves.cvmcriterion(distrib, x) ≈ 0.31140110078245337 # obtained with scipy.stats.cramervonmises
    end

    # Load test data for which the quantities have been computed manually
    df = CSV.read("data/702S006.csv", DataFrame)
    tags = names(df)[2:10]
    durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
    duration_dict = Dict(zip(tags, durations))
    data = IDFdata(df, "Year", duration_dict)

    tag_out = "5min"
    d_out = getduration(data, tag_out)

    train_data = IDFCurves.excludeduration(data, "5min")
    ℓ = length(getdata(data, tag_out))

    fm = IDFCurves.fit_mle(SimpleScaling, train_data, 1.)

    @testset "validation_cvm_statistic()" begin
        import IDFCurves.validation_cvm_statistic
        @test validation_cvm_statistic(SimpleScaling, data; tag_out = "5min") ≈ 3.3440 rtol=1e-4
        @test validation_cvm_statistic(SimpleScaling, data, fm; tag_out = "5min") ≈ 3.3440 rtol=1e-4
    end

    @testset "cvm_components" begin
        components = IDFCurves.compute_cvm_components(fm, train_data, d_out, ℓ)

        v = [-0.0310430, -0.0486335, -0.195152, -2.07987]

        A = [0.303977 -0.157451 0.620122 -2.4917
            -0.157451 0.517504 0.168475 -0.544275
            0.620122 0.168475 16.6242 3.49233
            -2.4917 -0.544275 3.49233 296.449]

        @test components.cdf_gradient(0.8) ≈ v rtol=1e-4
        @test Matrix(components.information_factor) ≈ A rtol=1e-4
    end


    @testset "approx_eigenvalues()" begin

        @testset "Brownian bridge kernel" begin
            # Brownian bridge kernel:
            # ρ(u, v) = min(u, v) - u*v
            #
            # It is obtained from cvmkernel by setting the correction term to zero.
            g(u) = [0.0]
            A = cholesky(Matrix{Float64}(I, 1, 1))
            components = IDFCurves.CvMComponents(g, A)

            λs = IDFCurves.approx_eigenvalues(components, 3, nquad = 100)

            @test length(λs) == 3
            @test issorted(λs; rev=true)
            @test all(λs .>= -sqrt(eps(Float64)))

            @test isapprox(λs[1], 1 / π^2, rtol=2e-3)
            @test isapprox(λs[2], 1 / (4π^2), rtol=3e-3)
            @test isapprox(λs[3], 1 / (9π^2), rtol=5e-3)
        end

        @testset "test_data" begin
            components = IDFCurves.compute_cvm_components(fm, train_data, d_out, ℓ)
            λ = IDFCurves.approx_eigenvalues(components, 5)
            @test λ ≈  [0.151435, 0.0272948, 0.0117955, 0.00659074, 0.00426458] rtol=1e-4
        end
    end


    @testset "zolotarev_approx()" begin

        λs = [0.0, -1.0]
        @test_throws ArgumentError IDFCurves.zolotarev_approx(λs, 1.0)

        λs = [1.0]
        x = Distributions.quantile(Distributions.Chisq(1), 0.99)
        @test isapprox(
            IDFCurves.zolotarev_approx(λs, x),
            0.99;
            rtol=1e-2,
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
            rtol=1e-2,
        )

        λs = Float64[]
        for k in 1:10
            append!(λs, fill(1 / 2^k, 2k^2))
        end

        # The value 16.51 was obtained by simulation.
        @test isapprox(
            IDFCurves.zolotarev_approx(λs, 16.51),
            0.99;
            rtol=1e-2,
        )

    end

    @testset "scalingtest()" begin

        @test_throws ArgumentError scalingtest(SimpleScaling, data, tag_out="1min")

        test_struct = scalingtest(SimpleScaling, data, tag_out="5min", q=20)

        @test test_struct.test_statistic ≈ 3.3440 atol=1e-4

    end

    @testset "_pseudoobs_matrix()" begin

        new_tags = ["5min", "10min"]
        new_durations = Dict("5min" => 5/60, "10min" => 10/60)
        new_years = Dict("5min" => collect(2000:2002), "10min" => collect(2000:2002))
        new_obs = Dict("5min" => [3.0, 1.0, 2.0], "10min" => [10.0, 20.0, 20.0])
        new_data = IDFdata(new_tags, new_durations, new_years, new_obs)

        u = IDFCurves._pseudoobs_matrix(new_data, ["5min", "10min"])

        @test u[:, 1] ≈ [3/4, 1/4, 2/4]
        @test u[:, 2] ≈ [1/4, 2.5/4, 2.5/4]

    end

    @testset "_idfdata_from_pseudoobs()" begin

        import IDFCurves: _idfdata_from_pseudoobs

        U = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]

        fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

        new_data = _idfdata_from_pseudoobs(data, fd, U)

        @test gettag(new_data) == tags

        for tag in tags
            @test getduration(new_data, tag) == getduration(data, tag)
            @test getyear(new_data, tag) == collect(1:size(U, 1))
        end

        for (j, tag) in enumerate(tags)
            d = getduration(data, tag)
            pd = getdistribution(fd, d)

            expected = [Float64(quantile(pd, U[i, j])) for i in axes(U, 1)]

            @test getdata(new_data, tag) ≈ expected
        end
    end

end