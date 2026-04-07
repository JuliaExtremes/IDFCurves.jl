using Pkg
Pkg.activate(".")

using Distributions, Test

using IDFCurves


    @testset "rand(::UniversalScaling)" begin
        
        pd = UniversalScaling(60, 100, 1, .1, .8, 5, 10)

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


pd = UniversalScaling(60, 100, 1, .1, .8, 5, 10)

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



# TODO Change parameter name
# \alpha :exponent
# \delta : small-scale offset
# \tau : large-scale offset