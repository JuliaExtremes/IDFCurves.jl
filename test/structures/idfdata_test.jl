@testset "IDFdata" begin
    
    tags = ["30min", "1h", "24h"]
    durations = [.5, 1, 24]
    years = [2020, 2021]
    y = hcat(1:2, 3:4, 5:6)

    d1 = Dict("30min" => .5, "1h" => 1. , "24h" => 24.)
    d2 = Dict("30min" => years, "1h" => years , "24h" => years)
    d3 = Dict("30min" => y[:,1], "1h" => y[:,2] , "24h" => y[:,3])

    s = IDFdata(tags, d1, d2, d3)

    @testset "Construction" begin

        @test getdata(s) == d3
        @test getduration(s) == d1
        @test getyear(s) == d2
        @test gettag(s) == tags

    end

    @testset "get properties" begin

        for i in eachindex(tags)
            @test getdata(s, tags[i]) == y[:,i]
        end

        for i in eachindex(tags)
            @test getdata(s, tags[i], 2020) == y[1,i]
            @test getdata(s, tags[i], 2021) == y[2,i]
        end

        for i in eachindex(tags)
            @test getduration(s, tags[i]) == durations[i]
            @test getyear(s, tags[i]) == years
        end
 
        
        # @test getdata(s, "30min") == y[:,1]
        # @test getdata(s, "1h") == y[:,2]
        # @test getdata(s, "24h") == y[:,3]

        # @test getdata(s, "30min", 2020) == y[1,1]
        # @test getdata(s, "30min", 2021) == y[2,1]

        # @test getdata(s, "1h", 2020) == y[1,2]
        # @test getdata(s, "1h", 2021) == y[2,2]

        # @test getdata(s, "24h", 2020) == y[1,3]
        # @test getdata(s, "24h", 2021) == y[2,3]

        # @test getduration(s, "30min") == durations[1]


        # for tag in gettag(s)
        #     @test getduration(s, tag) == durations
        #     @test getyear(s, tag) == years
        # end

    end
    
end