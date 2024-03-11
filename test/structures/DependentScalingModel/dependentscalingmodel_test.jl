@testset "DependentScaling" begin
    
    @testset "DependentScalingModel construction" begin
        
        d = [0. 1.; 1. 0.]
        pd = GeneralScaling(1, 1, 1, 0, .8, .5)
        Σ = MaternCorrelationStructure(10., 1.)
        C = GaussianCopula

        dm = DependentScalingModel(pd, Σ, C)

        @test getmarginalmodel(dm) == pd
        @test getcopulatype(dm) == C
        @test getcorrelogram(dm) == Σ

    end

    @testset "get type of DependentScalingModel" begin
        obj = DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}

        @test IDFCurves.getmarginaltype(obj) == GeneralScaling
        @test IDFCurves.getcopulatype(obj) == GaussianCopula
        @test IDFCurves.getcorrelogramtype(obj) == MaternCorrelationStructure

    end

    @testset "loglikelihood(::DependentScalingModel)" begin
        
        tags = ["1h", "2h"]
        durations = [1., 2.]
        years = [2020, 2021]
        y = hcat(2:3, 0:1)

        d1 = Dict(zip(tags, durations))
        d2 = Dict(tags[1] => years, tags[2] => years)
        d3 = Dict(tags[1] => y[:,1], tags[2] => y[:,2])

        data = IDFdata(tags, d1, d2, d3)

        mm = GeneralScaling(1, 0, 1, 0, .8, .5)

        Σ = MaternCorrelationStructure(1., 1.)
        h = IDFCurves.logdist(durations)
        C = GaussianCopula(cor.(Σ, h))

        pd = DependentScalingModel(mm, Σ, GaussianCopula)

        # TODO: Verify this value
        @test loglikelihood(pd, data) ≈ -7.090619315218428

        # TODO : tests for other marginal model, correlation structure, and copula type.

    end

    @testset "fitting a dependent scaling model" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
            
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
                
        data = IDFdata(df, "Year", duration_dict)

        @testset "construct_model(Type{<:DependentScalingModel})" begin

            abstract_model = DependentScalingModel{SimpleScaling, MaternCorrelationStructure, GaussianCopula}
            model = IDFCurves.construct_model(abstract_model, data, 1, [IDFCurves.map_to_real_space(SimpleScaling, [20, 5, .04, .76]); [0., 0.]])
            @test typeof(getmarginalmodel(model)) <: SimpleScaling
            @test all( [params(getmarginalmodel(model))...] .≈ [20, 5, .04, .76] )
            @test IDFCurves.getcopulatype(model) == GaussianCopula
            @test typeof(getcorrelogram(model)) <: MaternCorrelationStructure
            @test all( [params(getcorrelogram(model))...] .≈ [1., 1.] )

            @test_throws AssertionError IDFCurves.construct_model(abstract_model, data, 1, [0.,0.,0.,0.,0.,0.,0.,0.])

            abstract_model = DependentScalingModel{GeneralScaling, ExponentialCorrelationStructure, TCopula}
            model = IDFCurves.construct_model(abstract_model, data, 1, [IDFCurves.map_to_real_space(GeneralScaling, [20, 5, .04, .76,.7]); [0.0]])
            @test typeof(getmarginalmodel(model)) <: GeneralScaling
            @test all( [params(getmarginalmodel(model))...] .≈ [20, 5, .04, .76, .7] )
            @test IDFCurves.getcopulatype(model) == TCopula
            @test typeof(getcorrelogram(model)) <: ExponentialCorrelationStructure
            @test all( [params(getcorrelogram(model))...] .≈ [1.] )

        end

        @testset "fit_mle(::DependentScalingModel)" begin

            abstract_model = DependentScalingModel{SimpleScaling, ExponentialCorrelationStructure, GaussianCopula}
            @test_throws AssertionError IDFCurves.fit_mle(abstract_model, data, 1, [0, 1, -0.1, .76, 1])

            fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .04, .76, 1])
            @test [params(getmarginalmodel(fd))...] ≈ [17.026, 5.0642, -0.0116, 0.68445] rtol = .1
            @test [params(getcorrelogram(fd))...] ≈ [4.2134] rtol = .1
            fd2 = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .0, .76, 1])
            @test shape(getmarginalmodel(fd2)) != 0.0

            abstract_model = DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}
            fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .04, .7, .1, 1., 1.])
            @test [params(getmarginalmodel(fd))...] ≈ [20.0113, 5.7805, -0.0243, 0.7685, 0.0714] rtol = .1
            @test [params(getcorrelogram(fd))...] ≈ [1.0056, 2.2939] rtol = .1

        end

        @testset "hessian(::DependentScalingModel)" begin
            #TODO
        end

        @testset "quantilevar(::DependentScalingModel)" begin
            #TODO
        end

        @testset "quantilecint(::DependentScalingModel)" begin
            #TODO
        end

    end

end