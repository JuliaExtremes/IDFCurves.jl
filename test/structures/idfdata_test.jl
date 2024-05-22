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

        @test gettag(s, .5) == "30min"
        @test_throws ErrorException gettag(s, 5)

    end

    @testset "excludeduration(::IDFdata, d)" begin
        
        s2 = IDFCurves.excludeduration(s, 1)
        @test getdata(s2) == Dict("30min" => y[:,1], "24h" => y[:,3])
        @test getduration(s2) == Dict("30min" => .5, "24h" => 24.)
        @test getyear(s2) == Dict("30min" => years, "24h" => years)
        @test gettag(s2) == ["30min", "24h"]

    end

    @testset "selectduration(::IDFdata, d)" begin
        
        s3 = IDFCurves.selectduration(s, 24)
        @test getdata(s3) == Dict("24h" => y[:,3])
        @test getduration(s3) == Dict("24h" => 24.)
        @test getyear(s3) == Dict("24h" => years)
        @test gettag(s3) == ["24h"]

    end

    @testset "Base.show(io, IDFdata)" begin
        # print IDFdata does not throw
        buffer = IOBuffer()
        @test_logs Base.show(buffer, s)
    
    end

    @testset "IDFdata(::DataFrame)" begin

        df = CSV.read(joinpath("..","data","702S006.csv"), DataFrame)
    
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
    
        data = IDFdata(df, "Year", duration_dict)
    
        @test duration_dict == getduration(data)
        for tag in gettag(data)
            @test getyear(data, tag) == df[:, :Year]
            @test getdata(data, tag) ≈ df[:, tag]
        end

        # toy dataframe to check how missing data is handled :

        durations = [1., 2.]
        tags = ["firstduration", "secondduration"]
        duration_dict = Dict(zip(tags, durations))
        df = DataFrame(Year = [2020, 2021, 2022], firstduration = [2, missing, 3], secondduration = [3, 1, missing])

        data = IDFdata(df, "Year", duration_dict)
        @test getdata(data, "firstduration") ≈ [2., 3.]
        @test getdata(data, "secondduration") ≈ [3., 1.]
    
    end

    df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
    tags = names(df)[2:10]
    durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
    duration_dict = Dict(zip(tags, durations))
    s = IDFdata(df, "Year", duration_dict)

    @testset "getKendalldata(::IDFdata)" begin
        
        kendall_data = IDFCurves.getKendalldata(s)

        @test names(kendall_data) == ["tag1", "tag2", "distance", "kendall"]
        @test size(kendall_data, 1) ≈ length(gettag(s)) * ( length(gettag(s)) - 1 ) / 2

        row = kendall_data[3,:]
        d1, d2 = getduration(s, row[:tag1]), getduration(s, row[:tag2])
        @test row[:distance] ≈ IDFCurves.logdist(d1, d2)
        y1, y2 = getdata(s, row[:tag1]), getdata(s, row[:tag2])
        @test row[:kendall] ≈ IDFCurves.corkendall(y1, y2)
    
    end
    
end

