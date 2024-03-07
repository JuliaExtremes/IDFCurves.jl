@testset "GeneralScaling" begin

    @testset "GeneralScaling construction" begin
        pd = GeneralScaling(60, 100, 1, 0, .8, 5)
        @test location(pd) == 100
        @test scale(pd) == 1
        @test shape(pd) == 0
        @test exponent(pd) ≈ .8 
        @test offset(pd) == 5
        @test duration(pd) == 60
        @test all([params(pd)...] .≈ [100, 1, 0, .8, 5])
        @test params_number(GeneralScaling) == 5
    end

    @testset "getdistribution(::GeneralScaling)" begin
        pd = GeneralScaling(60, 100, 1, 0, .8, 5)
        
        md = getdistribution(pd, 3*60)
        
        @test location(md) ≈ 43.31051048132165
        @test scale(md) ≈ 0.4331051048132165
        @test shape(md) ≈ 0.
        
    end

    @testset "map_to_param_space(::Type{<:GeneralScaling}, θ)" begin
        
        θ = [1., 0., 0., 0., 0.]
        @test IDFCurves.map_to_param_space(GeneralScaling, θ) ≈ [1., 1., 0., .5, 1.]
    end

    @testset "map_to_real_space(::Type{<:GeneralScaling}, θ)" begin
        
        θ = [1., 1., 0., .5, 1.]
        @test IDFCurves.map_to_real_space(GeneralScaling, θ) ≈ [1., 0., 0., 0., 0.]
    end

    @testset "Base.show(io, GeneralScaling)" begin
        # print GeneralScaling does not throw
        pd = GeneralScaling(60, 100, 1, 0, .8, 5)
        buffer = IOBuffer()
        @test_logs Base.show(buffer, pd)

    end

    @testset "cdf(::GeneralScaling)" begin
        pd = GeneralScaling(1, 100, 1, 0, .8, 5)

        @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1 , 0), 100)
        @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1 , 0), [100, 200])
    end

    @testset "loglikelihood(::GeneralScaling)" begin
        
        pd = GeneralScaling(3, 1, 1, .1, .5, 1)
        data = rand(pd, [1, 3], 3, tags=["1", "3"])
        y₁ = getdata(data, "1")
        y₃ = getdata(data, "3")

        ll = sum(logpdf.(GeneralizedExtremeValue(sqrt(2),sqrt(2),.1), y₁)) + sum(logpdf.(GeneralizedExtremeValue(1,1,.1), y₃))

        @test loglikelihood(pd, data) ≈ ll
    end

    @testset "quantile(::GeneralScaling)" begin
        pd = GeneralScaling(60, 100, 1, 0, .8, 5)
        
        @test quantile(pd, 60, .9) ≈ quantile(GeneralizedExtremeValue(100,1,0), .9)
        
    end

    @testset "rand(::GeneralScaling)" begin
        
        pd = GeneralScaling(60, 100, 1, .1, .8, 5)

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

    df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        
    tags = names(df)[2:10]
    durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
    duration_dict = Dict(zip(tags, durations))
            
    data = IDFdata(df, "Year", duration_dict)

    @testset "fit_mle(::GeneralScaling)" begin

        fd = IDFCurves.fit_mle_gradient_free(GeneralScaling, data, 1, [20, 5, .04, .76, .07])
        @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1
        fd2 = IDFCurves.fit_mle_gradient_free(GeneralScaling, data, 1, [20, 5, .0, .76, .07])
        @test [params(fd2)...] ≈ [params(fd)...] rtol=.1
        @test shape(fd2) != 0.0

        fd = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .04, .76, .07])
        @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1
        fd2 = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .0, .76, .07])
        @test [params(fd2)...] ≈ [params(fd)...] rtol=.1
        @test shape(fd2) != 0.0

        @testset "hessian(::GeneralScaling, data)" begin
        
            @test IDFCurves.hessian(fd, data) ≈ [21.5015 -10.5403 46.7217 -100.863 -283.946;
                -10.5403 37.1912 21.0899 -43.1767 31.1629;
                46.7217 21.0899 1332.12 412.91 -1281.44;
                -100.863 -43.1767 412.91 21878.6 -15727.8;
                -283.946 31.1629 -1281.44 -15727.8 23588.0] rtol=.05

        end

        @testset "quantilevar" begin
            @test IDFCurves.quantilevar(fd, data, 24, .95) ≈ 0.014404245774623275
        end

        @testset "quantilevar" begin
            @test quantilecint(fd, data, 24, .95) ≈ [3.2642, 3.7346] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, .1) ≈ [3.3020, 3.6968] atol = 1e-4
        end
        
    end

end