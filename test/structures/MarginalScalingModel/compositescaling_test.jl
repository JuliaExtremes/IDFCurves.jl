@testset "CompositeScaling" begin

    @testset "CompositeScaling construction" begin
        pd = CompositeScaling(60, 100, 1, 0, .8, .6)
        @test location(pd) == 100
        @test scale(pd) == 1
        @test shape(pd) == 0
        @test location_exponent(pd) ≈ .8 
        @test scale_exponent(pd) == .6
        @test duration(pd) == 60
        @test all([params(pd)...] .≈ [100, 1, 0, .8, .6])
        @test params_number(CompositeScaling) == 5
    end

    @testset "getdistribution(::CompositeScaling)" begin
        pd = CompositeScaling(60, 100, 1, 0, .8, .6)

        md = getdistribution(pd, 3*60)
        
        @test location(md) ≈ 41.52436465385057
        @test scale(md) ≈ 0.5172818579717865
        @test shape(md) ≈ 0.
        
    end

    @testset "construct_model(::Type{<:CompositeScaling}, θ)" begin

        θ = [1., 0., 0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(CompositeScaling, 1, θ) 
        
        θ = [1., 0., 0., 0., 0.]
        pd = IDFCurves.construct_model(CompositeScaling, 1, θ)
        @test pd isa CompositeScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5, .5])

    end

    @testset "construct_model(::Type{<:CompositeScaling}, θ, c)" begin

        θ = [0., 0., 0., 0.]
        c = [1., 1., .5, .5]
        @test_throws AssertionError IDFCurves.construct_model(CompositeScaling, 1, θ, c) 
        
        θ = [1., 1., 1., 1., 1.]
        c = [1., 0., 0., 0., 0.]
        pd = IDFCurves.construct_model(CompositeScaling, 1, θ, c)
        @test pd isa CompositeScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5, .5])

    end

    @testset "map_to_real_space(::Type{<:CompositeScaling}, θ)" begin

        @test_throws AssertionError IDFCurves.map_to_real_space(CompositeScaling, [1., 1., 0., 0., 1])
        @test_throws AssertionError IDFCurves.map_to_real_space(CompositeScaling, [1., 1., 0., 0.5, -0.1])
        
        θ = [1., 1., 0., .5, .5]
        @test IDFCurves.map_to_real_space(CompositeScaling, θ) ≈ [1., 0., 0., 0., 0.]
        
    end

    @testset "Base.show(io, CompositeScaling)" begin
        # print CompositeScaling does not throw
        pd = CompositeScaling(60, 100, 1, 0, .8, .5)
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pd)

    end

    @testset "cdf(::CompositeScaling)" begin
        pd = CompositeScaling(1, 100, 1, 0, .8, .5)

        @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1 , 0), 100)
        @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1 , 0), [100, 200])
    end

    @testset "loglikelihood(::CompositeScaling)" begin
        
        pd = CompositeScaling(3, 1, 1, .1, .5, .1)
        data = rand(pd, [1, 3], 3, tags=["1", "3"])
        y₁ = getdata(data, "1")
        y₃ = getdata(data, "3")

        μ₁ = 1 * (1/3)^(-0.5)  # ≈ 1.732 (√3)
        σ₁ = 1 * (1/3)^(-0.1)  # ≈ 1.116
        μ₃ = 1 * (3/3)^(-0.5)  # = 1
        σ₃ = 1 * (3/3)^(-0.1)  # = 1

        ll = sum(logpdf.(GeneralizedExtremeValue(μ₁, σ₁, .1), y₁)) + 
            sum(logpdf.(GeneralizedExtremeValue(μ₃, σ₃, .1), y₃))

        @test loglikelihood(pd, data) ≈ ll
    end

    @testset "quantile(::CompositeScaling)" begin
        pd = CompositeScaling(60, 100, 1, 0, .8, .5)
        
        @test quantile(pd, 60, .9) ≈ quantile(GeneralizedExtremeValue(100,1,0), .9)
        
    end

    @testset "rand(::CompositeScaling)" begin
        
        pd = CompositeScaling(60, 100, 1, .1, .8, .5)

        n = 1
        d = [.5, 1, 24]
        tag = ["1", "2", "3"]
        data = rand(pd, d)

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == collect(1:n) 
            @test length(getdata(data, tag[i])) == n
        end

        n = 3
        d = [.5, 1, 24]
        tag = ["1", "2", "3"]
        data = rand(pd, d, n)

        @test issetequal(gettag(data), tag)
        for i in eachindex(tag)
            @test getduration(data, tag[i]) ≈ d[i]
            @test getyear(data, tag[i]) == collect(1:n)
            @test length(getdata(data, tag[i])) == n
        end

        n = 3
        d = [.5, 1, 24]
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

    @testset "fitting a composite scaling model" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        @testset "initialize(::CompositeScaling)" begin
            
            init_vector = initialize(CompositeScaling, data, 1)
            @test length(init_vector) == 5
            @test init_vector[5] ≈ 0.001

            init_vector_SS = initialize(SimpleScaling, data, 1)
            @test init_vector[1:4] ≈ init_vector_SS
            
        end

        fd = IDFCurves.fit_mle(CompositeScaling, data, 1, [20, 5, .04, .76, .7])
        H = IDFCurves.hessian(fd, data)

        @testset "fit_mle(::CompositeScaling, data, d₀, initialvalues)" begin

            @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1
            fd2 = IDFCurves.fit_mle(CompositeScaling, data, 1, [20, 5, .0, .76, .7])
            @test [params(fd2)...] ≈ [params(fd)...] rtol=.1
            @test shape(fd2) != 0.0

        end

        @testset "fit_mle(::CompositeScaling, data, d₀)" begin

            fd3 = IDFCurves.fit_mle(CompositeScaling, data, 1)
            @test [params(fd3)...] ≈ [params(fd)...] rtol=.1

        end

        @testset "hessian(::CompositeScaling, data)" begin

            @test H ≈ [26.2926 -13.6271 49.8782 -286.8681 23.3926;
                       -13.6271 42.1008 14.9248 80.9137 -18.2866;
                       49.8782 14.9248 1232.9155 -151.9252 -352.9037;
                       -286.8681 80.9137 -151.9252 41047.2741 -8168.9697;
                       23.3926 -18.2866 -352.9037 -8168.9697 4719.1376] rtol=.05

        end

        @testset "quantilevar" begin
            @test IDFCurves.quantilevar(fd, data, 24, .95, H) ≈ 0.01793935826089695
            @test IDFCurves.quantilevar(fd, data, 24, .95) ≈ 0.01793935826089695
        end

        @testset "quantilecint" begin
            @test quantilecint(fd, data, 24, .95) ≈ [3.2809, 3.8059] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, y=0, α=.1) ≈ [3.3231, 3.7637] atol = 1e-4

            @test quantilecint(fd, data, 24, .95, H) ≈ [3.2809, 3.8059] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, H, 0, .1) ≈ [3.3231, 3.7637] atol = 1e-4
        end
        
    end

        @testset "fitting a composite scaling non-stationary model" begin
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

        pd = CompositeScaling(
            1.0,
            Covariates([Covariate("t", rcp_filtered[:, Symbol("Year")])], BaseParamComputation),
            Covariates(),
            Covariates(),
            Covariates(),
            Covariates(),
        )

        init_values = [18.1366, 1.0, 5.2874, 0.0486, 0.6942, 0.5]
            
        Σ = UncorrelatedStructure()
        C = IdentityCopula
        dsm = DependentScalingModel(pd, Σ, C) 
        fd = IDFCurves.fit_mle(dsm, data, 1,  init_values)

        @testset "non-stationary model construction" begin
            @test pd isa CompositeScaling
            @test duration(pd) == 1.0
            @test params_number(pd) == 6  # 5 base + 1 time covariate

            # Check that location parameter is time-varying
            @test length(pd.μ₀.covariate) == 1  # One time covariate
            @test pd.μ₀.covariate[1].name == "t"
        end

        @testset "model fitting with initial values" begin
            @test [params(fd)...] ≈ [[18.3145, 0.0365], [5.3700], [0.0], [0.6992], [0.7458]] rtol=.1

            fd2 = IDFCurves.fit_mle(dsm, data, 1, [20, 1, 5, .0, .76, 0.5])
            @test [params(fd2)...] ≈ [[18.3145, 0.0365], [5.3700], [0.0], [0.6992], [0.7458]] rtol=.1
        end

        @testset "quantile computation" begin
            q_1950 = quantile(getmarginalmodel(fd), 24, 0.95, 1)
            q_2000 = quantile(getmarginalmodel(fd), 24, 0.95, 50)
        
            @test eltype(q_1950) <: Real
            @test eltype(q_2000) <: Real
            @test all(isfinite, q_1950)
            @test all(isfinite, q_2000)
            
            # Due to positive time trend, quantiles should increase over time
            @test all(q_2000 .> q_1950)
        end

        @testset "hessian and parameter confidence intervals" begin
            H = IDFCurves.hessian(fd, data)
            @test size(H) == (6, 6)
            
            param_cints = parametercint(fd, data)
            @test param_cints ≈ [
                [17.2185, 19.0333], 
                [-0.6389, 0.7586], 
                [4.5753, 5.9038], 
                [0.0044, 0.1340], 
                [0.6805, 0.7189], 
                [0.7021, 0.8039]] rtol=.05
        end
    end

end