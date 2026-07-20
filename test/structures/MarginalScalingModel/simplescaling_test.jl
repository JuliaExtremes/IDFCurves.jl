
@testset "SimpleScaling" begin

    @testset "SimpleScaling construction" begin
        pd = SimpleScaling(2, 100, 1, 0.1, 0.8)
        @test location(pd) == 100
        @test scale(pd) == 1
        @test shape(pd) ≈ 0.1
        @test exponent(pd) ≈ 0.8
        @test duration(pd) == 2
        @test all([params(pd)...] .≈ [100, 1, 0.1, 0.8])
        @test params_number(SimpleScaling) == 4
    end

    @testset "getdistribution(::SimpleScaling)" begin
        pd = SimpleScaling(1, 100, 4, 0.1, 0.8)

        md = getdistribution(pd, 3*1)

        @test location(md) ≈ 41.52436465385057
        @test scale(md) ≈ 1.6609745861540228
        @test shape(md) ≈ 0.1

    end

    @testset "construct_model(::Type{<:SimpleScaling}, θ)" begin

        θ = [1., 0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(SimpleScaling, 1, θ)

        θ = [1., 0., 0., 0.]
        pd = IDFCurves.construct_model(SimpleScaling, 1, θ)
        @test pd isa SimpleScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈ [1., 1., 0., 0.5])

    end

    @testset "map_to_real_space(::Type{<:SimpleScaling}, θ)" begin

        @test_throws AssertionError IDFCurves.map_to_real_space(SimpleScaling, [1., 0., 0., 0.5])
        @test_throws AssertionError IDFCurves.map_to_real_space(SimpleScaling, [1., 1., 0., -0.1])

        θ = [1., 1., 0., 0.5]
        @test IDFCurves.map_to_real_space(SimpleScaling, θ) ≈ [1., 0., 0., 0.]
    end

    @testset "Base.show(io, SimpleScaling)" begin
        # print dGEV does not throw
        pd = SimpleScaling(1, 100, 1, 0, 0.8)
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pd)

    end

    @testset "cdf(::SimpleScaling)" begin
        pd = SimpleScaling(1, 100, 1, 0, 0.8)

        @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1, 0), 100)
        @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1, 0), [100, 200])
    end

    @testset "loglikelihood(::SimpleScaling)" begin

        pd = SimpleScaling(4, 2, 1, -0.1, 0.5)
        duration_dict = Dict(zip(["1", "4"], [1, 4]))
        data = rand(pd, duration_dict, 3)
        y₁ = getdata(data, "1")
        y₃ = getdata(data, "4")

        ll = sum(logpdf.(GeneralizedExtremeValue(4, 2, -0.1), y₁)) + sum(logpdf.(GeneralizedExtremeValue(2, 1, -0.1), y₃))

        @test loglikelihood(pd, data) ≈ ll
    end

    @testset "quantile(::SimpleScaling)" begin
        pd = SimpleScaling(2, 100, 35, 0, 0.8)

        @test quantile(pd, 2, 0.9) ≈ quantile(GeneralizedExtremeValue(100, 35, 0), 0.9)

    end

    @testset "rand(::SimpleScaling)" begin

        pd = SimpleScaling(4, 2, 1, -0.1, 0.8)

        n = 1
        d = [0.5, 1, 4]
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

    @testset "fitting a simple scaling model" begin

        import IDFCurves: initialize, initial_duration_offset

        df = CSV.read(joinpath("..", "data", "702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        fm = initialize(SimpleScaling, data, 1.)

        @testset "initialize(::SimpleScaling)" begin

            @test_throws ArgumentError initialize(SimpleScaling, data, -1.)
            @test_throws ArgumentError initialize(SimpleScaling, data, 1., 25.)

            @test fm isa SimpleScaling

        end

        @testset "initial_duration_offset()" begin

            δ̂ = initial_duration_offset(fm, data, 0.25)

            @test δ̂ > 0
            @test_throws ArgumentError initial_duration_offset(fm, data, 1 / 12)

        end

        @testset "fit_mle(::SimpleScaling, data, initialmodel)" begin

            fd = IDFCurves.fit_mle(SimpleScaling, data, fm)
            @test collect(params(fd)) ≈ [18.1366, 5.2874, 0.0486, 0.6942] rtol=0.1
            
        end

        @testset "fit_mle(::SimpleScaling, data, d₀)" begin

            fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)
            @test collect(params(fd)) ≈ [18.1366, 5.2874, 0.0486, 0.6942] rtol=0.1

        end

        fd = IDFCurves.fit_mle(SimpleScaling, data, fm)

        @testset "hessian(::SimpleScaling, data)" begin

            @test IDFCurves.hessian(fd, data) ≈ [24.2687 -12.2383 49.9538 -66.4114;
                -12.2383 41.7471 17.8326 -56.9225;
                49.9538 17.8326 1364.59 695.963;
                -66.4114 -56.9225 695.963 25166.9] rtol=0.05

        end

        @testset "quantilevar" begin
            @test IDFCurves.quantilevar(fd, data, 24, 0.95) ≈ 0.015413582108460257 atol = 1e-4
        end

        @testset "quantilecint" begin
            @test quantilecint(fd, data, 24, 0.95) ≈ [3.613737526616065, 4.100402261105994] atol = 1e-4
            @test quantilecint(fd, data, 24, 0.95, 0.1) ≈ [3.652858933876735, 4.061280853845323] atol = 1e-4
        end

    end

end