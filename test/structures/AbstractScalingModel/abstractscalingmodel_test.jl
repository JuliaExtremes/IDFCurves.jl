include("dGEV_test.jl")
include("SimpleScaling_test.jl")

# simple scaling

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

df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
        
data = IDFdata(df, "Year", duration_dict)

@testset "fit_mle(::SimpleScaling)" begin

    fd = IDFCurves.fit_mle_gradient_free(SimpleScaling, data, 1, [20, 5, .04, .76])

    @test [params(fd)...] ≈ [18.1366, 5.2874, 0.0486, 0.6942] rtol=.1

    fd = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76])

    @test [params(fd)...] ≈ [18.1366, 5.2874, 0.0486, 0.6942] rtol=.1

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
        @test quantilecint(fd, data, 24, .95, .1) ≈ [3.652858933876735, 4.061280853845323] atol = 1e-4
    end
    
end


## dGEV


@testset "cdf(::dGEV)" begin
    pd = dGEV(1, 100, 1, 0, .8, 5)

    @test cdf(pd, 1, 100) ≈ cdf(GeneralizedExtremeValue(100, 1 , 0), 100)
    @test cdf(pd, 1, [100, 200]) ≈ cdf.(GeneralizedExtremeValue(100, 1 , 0), [100, 200])
end

@testset "loglikelihood(::dGEV)" begin
    
    pd = dGEV(3, 1, 1, .1, .5, 1)
    data = rand(pd, [1, 3], 3, tags=["1", "3"])
    y₁ = getdata(data, "1")
    y₃ = getdata(data, "3")

    ll = sum(logpdf.(GeneralizedExtremeValue(sqrt(2),sqrt(2),.1), y₁)) + sum(logpdf.(GeneralizedExtremeValue(1,1,.1), y₃))

    @test loglikelihood(pd, data) ≈ ll
end

@testset "quantile(::dGEV)" begin
    pd = dGEV(60, 100, 1, 0, .8, 5)
    
    @test quantile(pd, 60, .9) ≈ quantile(GeneralizedExtremeValue(100,1,0), .9)
    
end

@testset "rand(::dGEV)" begin
    
    pd = dGEV(60, 100, 1, .1, .8, 5)

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

@testset "fit_mle(::dGEV)" begin

    fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [20, 5, .04, .76, .07])

    @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1

    fd = IDFCurves.fit_mle(dGEV, data, 1, [20, 5, .04, .76, .07])

    @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1

    @testset "hessian(::dGEV, data)" begin
    
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
