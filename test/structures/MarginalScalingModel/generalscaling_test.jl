@testset "GeneralScaling" begin

    @testset "GeneralScaling construction" begin
        pd = GeneralScaling(60, 100, 1, 0, 0.8, 5)
        @test location(pd) == 100
        @test scale(pd) == 1
        @test shape(pd) == 0
        @test exponent(pd) ≈ 0.8
        @test offset(pd) == 5
        @test duration(pd) == 60
        @test all([params(pd)...] .≈ [100, 1, 0, 0.8, 5])
        @test params_number(GeneralScaling) == 5
    end

    @testset "scaling_factor(sm::GeneralScaling, d::Real)" begin
        import IDFCurves.scaling_factor

        d₀, μ₀, σ₀, ξ, α, δ = (0.5, 1., 1., 0., 0.5, 0.5)
        pd = GeneralScaling(d₀, μ₀, σ₀, ξ, α, δ)

        @test_throws AssertionError scaling_factor(pd, -1)

        # No scaling
        @test scaling_factor(pd, d₀) ≈ 1.

        #Scaling
        @test scaling_factor(pd, 3.5) ≈ 0.5
    end

    @testset "getdistribution(::GeneralScaling)" begin
        pd = GeneralScaling(60, 100, 1, 0, 0.8, 5)

        md = getdistribution(pd, 3 * 60)

        @test location(md) ≈ 43.31051048132165
        @test scale(md) ≈ 0.4331051048132165
        @test shape(md) ≈ 0.

    end

    @testset "construct_model(::Type{<:GeneralScaling}, θ)" begin

        θ = [1., 0., 0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(GeneralScaling, 1, θ)

        θ = [1., 0., 0., 0., 0.]
        pd = IDFCurves.construct_model(GeneralScaling, 1, θ)
        @test pd isa GeneralScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈ [1., 1., 0., 0.5, 1.])

        θ = [1., 0., 0., 0., -Inf]
        @test offset(IDFCurves.construct_model(GeneralScaling, 1, θ)) ≈ 0.

    end

    @testset "map_to_real_space(::Type{<:GeneralScaling}, θ)" begin

        @test_throws AssertionError IDFCurves.map_to_real_space(GeneralScaling, [1., -1, 0., 0.5, 0.1])
        @test_throws AssertionError IDFCurves.map_to_real_space(GeneralScaling, [1., 1., 0., 0., 0.1])
        @test_throws AssertionError IDFCurves.map_to_real_space(GeneralScaling, [1., 1., 0., 0.5, -0.1])

        θ = [1., 1., 0., 0.5, 1.]
        @test IDFCurves.map_to_real_space(GeneralScaling, θ) ≈ [1., 0., 0., 0., 0.]

    end

    @testset "Base.show(io, GeneralScaling)" begin
        # print GeneralScaling does not throw
        pd = GeneralScaling(60, 100, 1, 0, 0.8, 5)
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pd)

    end

    @testset "cdf(::GeneralScaling)" begin
        pd = GeneralScaling(1, 100, 1, 0, 0.8, 5)

        @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1, 0), 100)
        @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1, 0), [100, 200])
    end

    @testset "loglikelihood(::GeneralScaling)" begin

        pd = GeneralScaling(3, 1, 1, 0.1, 0.5, 1)
        duration_dict = Dict(zip(["1", "3"], [1, 3]))
        data = rand(pd, duration_dict)
        y₁ = getdata(data, "1")
        y₃ = getdata(data, "3")

        ll = sum(logpdf.(GeneralizedExtremeValue(sqrt(2), sqrt(2), 0.1), y₁)) + sum(logpdf.(GeneralizedExtremeValue(1, 1, 0.1), y₃))

        @test loglikelihood(pd, data) ≈ ll
    end

    @testset "quantile(::GeneralScaling)" begin
        pd = GeneralScaling(60, 100, 1, 0, 0.8, 5)

        @test quantile(pd, 60, 0.9) ≈ quantile(GeneralizedExtremeValue(100, 1, 0), 0.9)

    end

    @testset "rand(::GeneralScaling)" begin

        pd = GeneralScaling(60, 100, 1, 0.1, 0.8, 5)

        n = 1
        d = [0.5, 1, 24]
        tag = ["1", "2", "3"]
        duration_dict = Dict(zip(tag, d))
        data = rand(pd, duration_dict)

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == collect(1:n)
            @test length(getdata(data, tag[i])) == n
        end

        n = 3
        data = rand(pd, duration_dict, n)

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == collect(1:n)
            @test length(getdata(data, tag[i])) == n
        end

    end

    @testset "fitting a general scaling model" begin

        df = CSV.read(joinpath("..", "data", "702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        fm = initialize(GeneralScaling, data, 1., .25)

        @testset "initialize(::GeneralScaling)" begin

            @test_throws ArgumentError initialize(GeneralScaling, data, -1.)
            @test_throws ArgumentError initialize(GeneralScaling, data, 1., 25.)

            @test fm isa GeneralScaling

        end

        @testset "fit_mle(::GeneralScaling, data, initialmodel)" begin

            fd = IDFCurves.fit_mle(GeneralScaling, data, fm)
            @test collect(params(fd)) ≈ [19.79114, 5.5938, 0.0405, 0.7609, 0.0681] rtol=0.1
            
        end

        @testset "fit_mle(::GeneralScaling, data, d₀)" begin

            fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)
            @test collect(params(fd)) ≈ [19.79114, 5.5938, 0.0405, 0.7609, 0.0681] rtol=0.1

        end

    end

end