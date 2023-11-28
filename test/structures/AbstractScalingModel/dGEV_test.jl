
@testset "dGEV construction" begin
    pd = dGEV(60, 100, 1, 0, .8, 5)
    @test location(pd) == 100
    @test scale(pd) == 1
    @test shape(pd) == 0
    @test exponent(pd) ≈ .8 
    @test offset(pd) == 5
    @test duration(pd) == 60
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