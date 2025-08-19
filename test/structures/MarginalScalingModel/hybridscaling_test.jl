@testset "HybridScaling" begin

    @testset "HybridScaling construction" begin
        pd = HybridScaling(60, 100, 1, 0, .8, .6)
        @test location(pd) == 100
        @test scale(pd) == 1
        @test shape(pd) == 0
        @test location_exponent(pd) ≈ .8 
        @test scale_exponent(pd) == .6
        @test duration(pd) == 60
        @test all([params(pd)...] .≈ [100, 1, 0, .8, .6])
        @test params_number(HybridScaling) == 5
    end

    @testset "getdistribution(::HybridScaling)" begin
        pd = HybridScaling(60, 100, 1, 0, .8, .6)

        md = getdistribution(pd, 3*60)
        
        @test location(md) ≈ 51.72818579717865
        @test scale(md) ≈ 0.5172818579717865
        @test shape(md) ≈ 0.
        
    end

    @testset "construct_model(::Type{<:HybridScaling}, θ)" begin

        θ = [1., 0., 0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(HybridScaling, 1, θ) 

        θ = [1., 0., 0., 0., 0.]
        pd = IDFCurves.construct_model(HybridScaling, 1, θ)
        @test pd isa HybridScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5, .5])

    end

    @testset "construct_model(::Type{<:HybridScaling}, θ, c)" begin

        θ = [0., 0., 0., 0.]
        c = [1., 1., .5, .5]
        @test_throws AssertionError IDFCurves.construct_model(HybridScaling, 1, θ, c) 
        
        θ = [1., 1., 1., 1., 1.]
        c = [1., 0., 0., 0., 0.]
        pd = IDFCurves.construct_model(HybridScaling, 1, θ, c)
        @test pd isa HybridScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5, .5])

    end

    @testset "map_to_real_space(::Type{<:HybridScaling}, θ)" begin

        @test_throws AssertionError IDFCurves.map_to_real_space(HybridScaling, [1., 1., 0., 0., 1])
        @test_throws AssertionError IDFCurves.map_to_real_space(HybridScaling, [1., 1., 0., 0.5, -0.1])
        
        θ = [1., 1., 0., .5, .5]
        @test IDFCurves.map_to_real_space(HybridScaling, θ) ≈ [1., 0., 0., 0., 0.]
        
    end

    @testset "Base.show(io, HybridScaling)" begin
        # print HybridScaling does not throw
        pd = HybridScaling(60, 100, 1, 0, .8, .5)
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pd)

    end

    @testset "cdf(::HybridScaling)" begin
        pd = HybridScaling(1, 100, 1, 0, .8, .5)

        @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1 , 0), 100)
        @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1 , 0), [100, 200])
    end

    @testset "quantile(::HybridScaling)" begin
        pd = HybridScaling(60, 100, 1, 0, .8, .5)
        
        @test quantile(pd, 60, .9) ≈ quantile(GeneralizedExtremeValue(100,1,0), .9)
        
    end

    @testset "rand(::HybridScaling)" begin

        pd = HybridScaling(60, 100, 1, .1, .8, .5)

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

    @testset "fitting a hybrid scaling model" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        @testset "initialize(::HybridScaling)" begin
            
            init_vector = initialize(HybridScaling, data, 1)
            @test length(init_vector) == 5
            @test init_vector[5] ≈ 0.001

            init_vector_SS = initialize(SimpleScaling, data, 1)
            @test init_vector[1:4] ≈ init_vector_SS
            
        end

        fd = IDFCurves.fit_mle(HybridScaling, data, 1, [20, 5, .04, .76, .7])
        H = IDFCurves.hessian(fd, data)

        @testset "fit_mle(::HybridScaling, data, d₀, initialvalues)" begin

            @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1
            fd2 = IDFCurves.fit_mle(HybridScaling, data, 1, [20, 5, .0, .76, .7])
            @test [params(fd2)...] ≈ [params(fd)...] rtol=.1
            @test shape(fd2) != 0.0

        end

        @testset "fit_mle(::HybridScaling, data, d₀)" begin

            fd3 = IDFCurves.fit_mle(HybridScaling, data, 1)
            @test [params(fd3)...] ≈ [params(fd)...] rtol=.1

        end

        @testset "hessian(::HybridScaling, data)" begin

            @test H ≈ [19.92830 -9.84369 44.94858 249.68175 -312.61994; 
            -9.84369 34.38346 19.57815 -18.15253 -29.36098; 
            44.94858 19.57815 1330.90560 1013.54097 -398.09302; 
            249.68175 -18.15253 1013.54097 9370.89643 -0.0; 
            -312.61994 -29.36098 -398.09302 -0.0 16097.53637] rtol=.05

        end

        @testset "quantilevar" begin
            @test IDFCurves.quantilevar(fd, data, 24, .95, H) ≈ 0.014893013166648653
            @test IDFCurves.quantilevar(fd, data, 24, .95) ≈ 0.014893013166648653
        end

        @testset "quantilecint" begin
            @test quantilecint(fd, data, 24, .95) ≈ [3.2723, 3.7506] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, y=0, α=.1) ≈ [3.3107, 3.7122] atol = 1e-4

            @test quantilecint(fd, data, 24, .95, H) ≈ [3.2723, 3.7506] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, H, 0, .1) ≈ [3.3107, 3.7122] atol = 1e-4
        end
        
    end

    @testset "fitting a hybrid scaling non-stationary model" begin
        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        rcp = CSV.read(joinpath("..", "data/covariates","RCPdata.csv"), DataFrame)
        rcp_filtered = rcp[2017 .>= rcp.Year .>= 1943, :]
        rcp_filtered = rcp_filtered[rcp_filtered.Year .!= 1994, :]
        rcp_filtered = rcp_filtered[rcp_filtered.Year .!= 1995, :]
        rcp_filtered = rcp_filtered[rcp_filtered.Year .!= 2003, :]

        pd = HybridScaling(
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
            @test pd isa HybridScaling
            @test duration(pd) == 1.0
            @test params_number(pd) == 6  # 5 base + 1 time covariate

            # Check that location parameter is time-varying
            @test length(pd.μ₀.covariate) == 1  # One time covariate
            @test pd.μ₀.covariate[1].name == "t"
        end

        @testset "model fitting with initial values" begin
            @test [params(fd)...] ≈ [[20.6392, -0.0190], [5.9139], [0.0], [0.6071], [0.7589]] rtol=.1

            fd2 = IDFCurves.fit_mle(dsm, data, 1, [20, 1, 5, .0, .76, 0.5])
            @test [params(fd2)...] ≈ [[20.6392, -0.0190], [5.9139], [0.0], [0.6071], [0.7589]] rtol=.1
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
            @test size(H) == (6, 6)
            
            param_cints = parametercint(fd, data)
            @test param_cints ≈ [
                [18.9591, 21.9757], 
                [-0.8814, 0.8565], 
                [4.9744, 6.6593], 
                [-0.0231, 0.1085], 
                [0.5727, 0.6409], 
                [0.7287, 0.7846]] rtol=.05
        end
    end

end