
@testset "dGEV construction" begin
    pd = dGEV(60, 100, 1, 0, .8, 5)
    @test location(pd) == 100
    @test scale(pd) == 1
    @test shape(pd) == 0
    @test exponent(pd) ≈ .8 
    @test offset(pd) == 5
    @test duration(pd) == 60
    @test all([params(pd)...] .≈ [100, 1, 0, .8, 5])
end

@testset "getdistribution(::dGEV)" begin
    pd = dGEV(60, 100, 1, 0, .8, 5)
    
    md = getdistribution(pd, 3*60)
    
    @test location(md) ≈ 43.31051048132165
    @test scale(md) ≈ 0.4331051048132165
    @test shape(md) ≈ 0.
    
end

@testset "loglikelihood(::dGEV)" begin
    
    pd = dGEV(3, 1, 1, .1, .5, 1)
    data = rand(pd, [1, 3], 3, tags=["1", "3"])
    y₁ = getdata(data, "1")
    y₃ = getdata(data, "3")

    ll = sum(logpdf.(GeneralizedExtremeValue(sqrt(2),sqrt(2),.1), y₁)) + sum(logpdf.(GeneralizedExtremeValue(1,1,.1), y₃))

    @test loglikelihood(pd, data) ≈ ll
end

@testset "map_to_real_space(::Type{<:dGEV}, θ)" begin
    
    θ = [1., 0., -.5, 0., 0.]
    @test IDFCurves.map_to_real_space(dGEV, θ) ≈ [1., 1., .5, .5, 1.]
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

## Fit 

@testset "fit_mle(::dGEV)" begin

    df = CSV.read(joinpath("..", "data","IDF_702S006.csv"), DataFrame)
    
    tags = names(df)[2:10]
    durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
    duration_dict = Dict(zip(tags, durations))
        
    data = IDFdata(df, "Year", duration_dict)

    fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [1., 1., .1, .8, .01])

    @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1

    fd = IDFCurves.fit_mle(dGEV, data, 1, [1., 1., .1, .8, .01])

    @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1
    
end