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

    @testset "construct_model(::Type{<:GeneralScaling}, θ)" begin

        θ = [1., 0., 0., 0.]
        @test_throws AssertionError IDFCurves.construct_model(GeneralScaling, 1, θ) 
        
        θ = [1., 0., 0., 0., 0.]
        pd = IDFCurves.construct_model(GeneralScaling, 1, θ)
        @test pd isa GeneralScaling
        @test duration(pd) == 1
        @test all([params(pd)...] .≈  [1., 1., 0., .5, 1.])

        θ = [1., 0., 0., 0., -Inf]
        @test offset(IDFCurves.construct_model(GeneralScaling, 1, θ)) == 0.
        @test_logs (:warn,) IDFCurves.construct_model(GeneralScaling, 1, θ, final_model = true)

        θ = [1., 0., 0., 0., log(1e-15)]
        pd = IDFCurves.construct_model(GeneralScaling, 1, θ, final_model = true)
        @test pd isa SimpleScaling
        @test all([params(pd)...] .≈  [1., 1., 0., .5])

    end

    @testset "map_to_real_space(::Type{<:GeneralScaling}, θ)" begin

        @test_throws AssertionError IDFCurves.map_to_real_space(GeneralScaling, [1., -1, 0., 0.5, 0.1])
        @test_throws AssertionError IDFCurves.map_to_real_space(GeneralScaling, [1., 1., 0., 0., 0.1])
        @test_throws AssertionError IDFCurves.map_to_real_space(GeneralScaling, [1., 1., 0., 0.5, -0.1])
        
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

    @testset "fitting a general scaling model" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        @testset "initialize(::GeneralScaling)" begin
            
            init_vector = initialize(GeneralScaling, data, 1)
            @test length(init_vector) == 5
            @test init_vector[5] ≈ 0.001

            init_vector_SS = initialize(SimpleScaling, data, 1)
            @test init_vector[1:4] ≈ init_vector_SS
            
        end

        fd = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .04, .76, .07])
        H = IDFCurves.hessian(fd, data)

        @testset "fit_mle(::GeneralScaling, data, d₀, initialvalues)" begin

            @test [params(fd)...] ≈ [19.7911, 5.5938, 0.0405, 0.7609, 0.0681] rtol=.1
            fd2 = IDFCurves.fit_mle(GeneralScaling, data, 1, [20, 5, .0, .76, 1e-9])
            @test [params(fd2)...] ≈ [params(fd)...] rtol=.01
            @test shape(fd2) != 0.0

        end
        
        # data for which δ is estimated almost equal to 0
        df = DataFrame(Year = 1:5,
        d2 = [115.66936090096817, 102.07530027790483, 110.23552647428488, 101.43374631705547, 114.72830627764733],
        d3 = [85.24581170746404, 65.12966286341307, 76.7475813220242, 75.79353124455645, 68.56626670369397],
        d4 = [34.319734364436265, 31.72203736891103, 41.73319790518032, 42.38698775756336, 48.960074714341765],
        d5 = [22.52177638650152, 17.26322717397859, 19.168603864047387, 23.452621217405845, 20.953617397406624],
        d6 = [11.824403876924093, 9.714806478882196, 12.317907264280239, 11.186164746121602, 8.668935072544908],
        d7 = [6.150480234167272, 6.864025692753223, 5.6060380841178965, 5.4371321871641705, 8.222157251560155],
        d8 = [2.6727983163545854, 3.824937599625118, 3.5706517287382065, 3.6231397223924047, 3.2366781273031098],
        d9 = [1.8296778639474427, 1.7248572090544316, 2.2044817450846086, 2.061289389224367, 2.078686647048528])
        tags = names(df)[2:9]
        durations = [1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data_bug = IDFdata(df, "Year", duration_dict)

        fd4 = IDFCurves.fit_mle(GeneralScaling, data_bug, 1)
        H2 = IDFCurves.hessian(fd4, data_bug)

        @testset "fit_mle(::GeneralScaling, data, d₀)" begin

            fd3 = IDFCurves.fit_mle(GeneralScaling, data, 1)
            @test [params(fd3)...] ≈ [params(fd)...] rtol=.01

        end

        @testset "hessian(::GeneralScaling, data)" begin
        
            @test H ≈ [21.5015 -10.5403 46.7217 -100.863 -283.946;
                -10.5403 37.1912 21.0899 -43.1767 31.1629;
                46.7217 21.0899 1332.12 412.91 -1281.44;
                -100.863 -43.1767 412.91 21878.6 -15727.8;
                -283.946 31.1629 -1281.44 -15727.8 23588.0] rtol=.01

            @test H2 ≈ [2.78981     0.828548    10.199    -40.9208;
                0.828548    7.27113     23.6169   -41.5045;
                10.199      23.6169     255.837   -452.005;
                -40.9208    -41.5045    -452.005   5410.18] rtol=.01

        end

        @testset "quantilevar" begin
            @test IDFCurves.quantilevar(fd, data, 24, .95, H) ≈ 0.014404245774623275
            @test IDFCurves.quantilevar(fd, data, 24, .95) ≈ 0.014404245774623275
        end

        @testset "quantilecint" begin
            @test quantilecint(fd, data, 24, .95) ≈ [3.2642, 3.7346] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, .1) ≈ [3.3020, 3.6968] atol = 1e-4

            @test quantilecint(fd, data, 24, .95, H) ≈ [3.2642, 3.7346] atol = 1e-4
            @test quantilecint(fd, data, 24, .95, H, .1) ≈ [3.3020, 3.6968] atol = 1e-4
        end
        
    end

end