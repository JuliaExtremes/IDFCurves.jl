@testset "SimpleScaling" begin

    @testset "SimpleScaling construction" begin
        pd = SimpleScaling(2, 100, 1, 0.1, .8)
        @test location(pd) == 100
        @test scale(pd) == 1
        @test shape(pd) ≈ 0.1
        @test exponent(pd) ≈ .8 
        @test duration(pd) == 2
        @test all([params(pd)...] .≈ [100, 1, 0.1, .8])
        @test params_number(SimpleScaling) == 4
    end

    @testset "getdistribution(::SimpleScaling)" begin
        pd = SimpleScaling(1, 100, 4, 0.1, .8)
        
        md = getdistribution(pd, 3*1)
        
        @test location(md) ≈ 41.52436465385057
        @test scale(md) ≈ 1.6609745861540228
        @test shape(md) ≈ 0.1
        
    end

    @testset "construct_model(::Type{<:SimpleScaling}, d₀, θ)" begin

        θ = [1., 0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(SimpleScaling, 1, θ)
        
        θ = [1., 0., 0., 0.]
        pd = IDFCurves.construct_model(SimpleScaling, 1, θ)
        @test pd isa SimpleScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5])

    end

    @testset "construct_model(::Type{<:SimpleScaling}, d₀, θ, c)" begin

        θ = [0., 0., 0.]
        c = [1., 1., 0.5]
        @test_throws AssertionError IDFCurves.construct_model(SimpleScaling, 1, θ, c)
        
        θ = [1., 1., 1., 1.]
        c = [1., 0., 0., 0.]
        pd = IDFCurves.construct_model(SimpleScaling, 1, θ, c)

        @test pd isa SimpleScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5])
    end

    @testset "map_to_real_space(::Type{<:SimpleScaling}, θ)" begin

        @test_throws AssertionError IDFCurves.map_to_real_space(SimpleScaling, [1., 0., 0., 0.5])
        @test_throws AssertionError IDFCurves.map_to_real_space(SimpleScaling, [1., 1., 0., -0.1])
        
        θ = [1., 1., 0., .5]
        @test IDFCurves.map_to_real_space(SimpleScaling, θ) ≈ [1., 0., 0., 0.]
    end

    @testset "Base.show(io, SimpleScaling)" begin
        pd = SimpleScaling(1, 100, 1, 0, .8)
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pd)

    end

    @testset "cdf(::SimpleScaling)" begin
        pd = SimpleScaling(1, 100, 1, 0, .8)

        @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1 , 0), 100)
        @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1 , 0), [100, 200])
    end

    @testset "loglikelihood(::SimpleScaling)" begin
        
        pd = SimpleScaling(4, 2, 1, -.1, .5)
        data = rand(pd, [1, 4], 3, tags=["1", "4"])
        y₁ = getdata(data, "1")
        y₃ = getdata(data, "4")

        ll = sum(logpdf.(GeneralizedExtremeValue(4,2,-.1), y₁)) + sum(logpdf.(GeneralizedExtremeValue(2,1,-.1), y₃))

        @test loglikelihood(pd, data) ≈ ll
    end

    @testset "quantile(::SimpleScaling)" begin
        pd = SimpleScaling(2, 100, 35, 0, .8)
        
        @test quantile(pd, 2, .9) ≈ quantile(GeneralizedExtremeValue(100,35,0), .9)
        
    end

    @testset "rand(::SimpleScaling)" begin
        
        pd = SimpleScaling(4, 2, 1, -.1, .8)

        n = 1
        d = [.5, 1, 4]
        tag = ["1", "2", "3"]
        data = rand(pd, d)

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == collect(1:n) 
            @test length(getdata(data, tag[i])) == n
        end

        n = 3
        d = [.5, 1, 4]
        tag = ["1", "2", "3"]
        data = rand(pd, d, n)

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == collect(1:n)
            @test length(getdata(data, tag[i])) == n
        end

        n = 3
        d = [.5, 1, 4]
        tag = ["10", "11", "12"]
        x = [10, 11, 12]
        data = rand(pd, d, n, tags = tag, x = [10, 11, 12])

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == x
            @test length(getdata(data, tag[i])) == n
        end

    end

    @testset "fitting a simple scaling model" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        @testset "initialize(::SimpleScaling)" begin
            
            init_vector = initialize(SimpleScaling, data, 24)
            @test length(init_vector) == 4
            @test init_vector ≈ [1.82042, 0.50181, 0.0, 0.7257] rtol=.1

            init_vector2 = initialize(SimpleScaling, data, 1)
            @test init_vector2[4] ≈ init_vector[4] 
            @test log(init_vector2[1]) ≈ log(24)*init_vector2[4] + log(init_vector[1])

            Random.seed!(37)
            df = DataFrame(Year = 1:20, d1 = rand(GeneralizedExtremeValue(50., 7., -0.2), 20), d2 = rand(GeneralizedExtremeValue(35., 5., -0.1), 20) )
            duration_dict = Dict(zip(["d1", "d2"], [1/12, 1/6]))
            data2 = IDFdata(df, "Year", duration_dict)

            @test initialize(SimpleScaling, data2, 1) isa Any

        end

        fd = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76])

        @testset "fit_mle(::SimpleScaling, data, d₀, initialvalues)" begin

            @test [params(fd)...] ≈ [18.1366, 5.2874, 0.0486, 0.6942] rtol=.1
            fd2 = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .0, .76])
            @test [params(fd2)...] ≈ [params(fd)...] rtol=.1
            @test shape(fd2) != 0.0

        end

        @testset "fit_mle(::SimpleScaling, data, d₀)" begin

            fd3 = IDFCurves.fit_mle(SimpleScaling, data, 1)
            @test [params(fd3)...] ≈ [params(fd)...] rtol=.1

        end

        @testset "hessian(::SimpleScaling, data)" begin
        
            @test IDFCurves.hessian(fd, data) ≈ [24.2687  -12.2383    49.9538    -66.4114;
            -12.2383   41.7471    17.8326    -56.9225;
                49.9538   17.8326  1364.59      695.963;
                -66.4114  -56.9225   695.963   25166.9] rtol=.05

        end

        @testset "quantilevar" begin
            @test IDFCurves.quantilevar(fd, data, 24, .95) ≈ 0.015413582108460257
        end

        @testset "quantilecint" begin
            @test quantilecint(fd, data, 24, .95) ≈ [3.613737526616065, 4.100402261105994] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, y=0, α=.1) ≈ [3.652858933876735, 4.061280853845323] atol = 1e-4
        end
        
    end

    @testset "fitting a simple scaling non-stationary model" begin
        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        rcp = CSV.read(joinpath("..", "data","RCPdata.csv"), DataFrame)
        rcp_filtered = rcp[2017 .>= rcp.Year .>= 1943, :]
        rcp_filtered = rcp_filtered[rcp_filtered.Year .!= 1994, :]
        rcp_filtered = rcp_filtered[rcp_filtered.Year .!= 1995, :]
        rcp_filtered = rcp_filtered[rcp_filtered.Year .!= 2003, :]

        pd = SimpleScaling(
            1.0,
            Covariates([Covariate("t", rcp_filtered[:, Symbol("Year")])], BaseParamComputation),
            Covariates(),
            Covariates(),
            Covariates(),
        )

        init_values = [18.1366, 1.0, 5.2874, 0.0486, 0.6942]
            
        Σ = UncorrelatedStructure()
        C = IdentityCopula
        dsm = DependentScalingModel(pd, Σ, C) 
        fd = IDFCurves.fit_mle(dsm, data, 1,  init_values)

        @testset "non-stationary model construction" begin
            @test pd isa SimpleScaling
            @test duration(pd) == 1.0
            @test params_number(pd) == 5  # 4 base + 1 time covariate

            # Check that location parameter is time-varying
            @test length(pd.μ₀.covariate) == 1  # One time covariate
            @test pd.μ₀.covariate[1].name == "t"
        end

        @testset "model fitting with initial values" begin
            @test [params(fd)...] ≈ [[18.1366, -0.0753], [5.2874], [0.0486], [0.6942]] rtol=.1

            fd2 = IDFCurves.fit_mle(dsm, data, 1, [20, 1, 5, .0, .76])
            @test [params(fd2)...] ≈ [[18.1366, -0.0753], [5.2874], [0.0486], [0.6942]] rtol=.1
        end

        @testset "quantile computation" begin
            q_1950 = quantile(getmarginalmodel(fd), 24, 0.95, 1)
            q_2000 = quantile(getmarginalmodel(fd), 24, 0.95, 50)
        
            @test eltype(q_1950) <: Real
            @test eltype(q_2000) <: Real
            @test all(isfinite, q_1950)
            @test all(isfinite, q_2000)
            
            # Due to negative time trend, quantiles should decrease over time
            @test all(q_2000 .< q_1950)
        end

        @testset "hessian and parameter confidence intervals" begin
            H = IDFCurves.hessian(fd, data)
            @test size(H) == (5, 5)
            
            param_cints = parametercint(fd, data)
            @test param_cints ≈ [
                [17.2287, 19.0434], 
                [-0.8339, 0.6834], 
                [4.61735, 5.9544], 
                [-0.015, 0.1129], 
                [0.6765, 0.7123]] rtol=.05
        end
    end
end