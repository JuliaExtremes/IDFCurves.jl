using IDFCurves
using CSV, DataFrames, Distributions, ForwardDiff, PDMats, Random, SpecialFunctions, Test

@testset "DependentScaling" begin

    @testset "get subtypes of DependentScalingModel" begin
        obj_type = DependentScalingModel{SimpleScaling, ExponentialCorrelationStructure, GaussianCopula}

        @test IDFCurves.getmarginaltype(obj_type) == SimpleScaling
        @test IDFCurves.getcopulatype(obj_type) == GaussianCopula
        @test IDFCurves.getcorrelogramtype(obj_type) == ExponentialCorrelationStructure

    end
    
    @testset "DependentScalingModel construction" begin
        
        pd = GeneralScaling(1, 1, 1, 0, .8, .5)
        Σ = MaternCorrelationStructure(10., 1.)
        C = GaussianCopula

        dm = DependentScalingModel(pd, Σ, C)

        @test getmarginalmodel(dm) == pd
        @test getcopulatype(dm) == C
        @test getcorrelogram(dm) == Σ
        @test duration(dm) == 1
        @test all([params(dm)...] .≈ [1, 1, 0, .8, .5, 10., 1.])

        @test IDFCurves.getabstracttype(dm) == DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}
        

    end

    @testset "quantile(::DependentScalingModel)" begin
        pd = GeneralScaling(1, 1, 1, 0, .8, .5)
        Σ = MaternCorrelationStructure(10., 1.)
        C = GaussianCopula

        dm = DependentScalingModel(pd, Σ, C)
        
        @test quantile(dm, 2, .9) ≈ quantile(pd, 2, .9)
        
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
        pd = DependentScalingModel(mm, Σ, GaussianCopula)
        @test loglikelihood(pd, data) ≈ -7.090619315218428
        pd = DependentScalingModel(mm, Σ, TCopula{10})
        @test loglikelihood(pd, data) ≈ -7.2149300395481

        mm = SimpleScaling(1, 0, 1, 0, .8)
        Σ = ExponentialCorrelationStructure(1.)
        pd = DependentScalingModel(mm, Σ, TCopula{3})
        @test loglikelihood(pd, data) ≈ -6.980377452138165

        mm = SimpleScaling(1, 0, 1, 0, .8)
        pd = DependentScalingModel(mm, UncorrelatedStructure(), IdentityCopula)
        @test loglikelihood(pd, data) ≈ -6.992515226105631
        pd2 = DependentScalingModel(mm, ExponentialCorrelationStructure(1.), IdentityCopula)
        pd3 = DependentScalingModel(mm, UncorrelatedStructure(), GaussianCopula)
        @test loglikelihood(pd, data) ≈ loglikelihood(pd2, data) ≈ loglikelihood(pd3, data)

    end

    @testset "fitting a dependent scaling model" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        @testset "initialize(Type{<:DependentScalingModel}, data, d₀)" begin
            
            abstract_model = DependentScalingModel{SimpleScaling, MaternCorrelationStructure, GaussianCopula}

            init_vector = initialize(abstract_model, data, 1)
            init_scaling_model = initialize(SimpleScaling, data, 1)
            init_corr_structure = initialize(MaternCorrelationStructure, data)

            @test length(init_vector) == length(init_scaling_model) + length(init_corr_structure)
            @test all( init_vector[1:length(init_scaling_model)] .≈ init_scaling_model )
            @test all( init_vector[(length(init_scaling_model)+1):end] .≈ init_corr_structure )

        end

        @testset "construct_model(Type{<:DependentScalingModel})" begin

            abstract_model = DependentScalingModel{SimpleScaling, MaternCorrelationStructure, GaussianCopula}
            model = IDFCurves.construct_model(abstract_model, 1, [IDFCurves.map_to_real_space(SimpleScaling, [20, 5, .04, .76]); [0., 0.]])
            @test typeof(getmarginalmodel(model)) <: SimpleScaling
            @test all( [params(getmarginalmodel(model))...] .≈ [20, 5, .04, .76] )
            @test IDFCurves.getcopulatype(model) == GaussianCopula
            @test typeof(getcorrelogram(model)) <: MaternCorrelationStructure
            @test all( [params(getcorrelogram(model))...] .≈ [1., 1.] )

            @test_throws AssertionError IDFCurves.construct_model(abstract_model, 1, [0.,0.,0.,0.,0.,0.,0.,0.])

            abstract_model = DependentScalingModel{GeneralScaling, ExponentialCorrelationStructure, TCopula{1}}
            model = IDFCurves.construct_model(abstract_model, 1, [IDFCurves.map_to_real_space(GeneralScaling, [20, 5, .04, .76,.7]); [0.0]])
            @test typeof(getmarginalmodel(model)) <: GeneralScaling
            @test all( [params(getmarginalmodel(model))...] .≈ [20, 5, .04, .76, .7] )
            @test IDFCurves.getcopulatype(model) == TCopula{1}
            @test typeof(getcorrelogram(model)) <: ExponentialCorrelationStructure
            @test all( [params(getcorrelogram(model))...] .≈ [1.] )

        end

        @testset "map_to_real_space(Type{<:DependentScalingModel})" begin

            abstract_model = DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}

            @test_throws AssertionError IDFCurves.map_to_real_space(abstract_model, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            @test_throws AssertionError IDFCurves.map_to_real_space(abstract_model, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -1])
            @test_throws AssertionError IDFCurves.map_to_real_space(abstract_model, [0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5])

            params_vector = [20, 5, .04, .76, .7, 1., 2.]
            θ = IDFCurves.map_to_real_space(abstract_model, params_vector)
            @test all( θ[1:5] .≈ IDFCurves.map_to_real_space(GeneralScaling, params_vector[1:5]) )
            @test all( θ[6:7] .≈ IDFCurves.map_to_real_space(MaternCorrelationStructure, params_vector[6:7]) )

        end

        abstract_model = DependentScalingModel{SimpleScaling, UncorrelatedStructure, IdentityCopula}
        init_vector = initialize(abstract_model, data, 1)
        fd = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .04, .76])

        @testset "fit_mle(::DependentScalingModel, data, d₀, initialvalues)" begin

            abstract_model = DependentScalingModel{SimpleScaling, ExponentialCorrelationStructure, GaussianCopula}
            @test_throws AssertionError IDFCurves.fit_mle(abstract_model, data, 1, [0, 1, 0.1, .76, 1, 1])
            @test_throws AssertionError IDFCurves.fit_mle(abstract_model, data, 1, [0, 1, -0.1, .76, 1])

            @test [params(getmarginalmodel(fd))...] ≈ [18.13658321683213, 5.287438529290354, 0.04856483747914808, 0.6942332103996621] rtol = .01

            abstract_model = DependentScalingModel{SimpleScaling, UncorrelatedStructure, IdentityCopula}
            fd2 = IDFCurves.fit_mle(abstract_model, data, 1, [20, 5, .0, .76])
            @test shape(getmarginalmodel(fd2)) != 0.0
            init_vector = initialize(abstract_model, data, 1)

            #TODO test when xi is initialized close to 0

        end

        @testset "fit_mle(::DependentScalingModel, data, d₀s)" begin

            abstract_model = DependentScalingModel{SimpleScaling, ExponentialCorrelationStructure, GaussianCopula}

            init_vector = initialize(abstract_model, data, 1)

            fd3 = IDFCurves.fit_mle(abstract_model, data, 1, init_vector)
            fd4 = IDFCurves.fit_mle(abstract_model, data, 1)

            @test [params(getmarginalmodel(fd3))...] ≈ [params(getmarginalmodel(fd4))...]

        end

        @testset "hessian(::DependentScalingModel)" begin

            pd = GeneralScaling(1, 1, 1, 0, .8, .5)
            Σ = MaternCorrelationStructure(10., 1.)
            C = GaussianCopula
            dm = DependentScalingModel(pd, Σ, C)

            H = IDFCurves.hessian(dm, data)
            @test size(H,1) == 7
            
            H = IDFCurves.hessian(fd,data)
            @test H ≈ [24.2687  -12.2383    49.9538    -66.4114;
                -12.2383   41.7471    17.8326    -56.9225;
                49.9538   17.8326  1364.59      695.963;
                -66.4114  -56.9225   695.963   25166.9] rtol=.01

        end

        @testset "quantilevar(::DependentScalingModel)" begin

            @test_throws AssertionError IDFCurves.quantilevar(fd, data, 0, 0.99)
            @test_throws AssertionError IDFCurves.quantilevar(fd, data, 1, 1)

            @test IDFCurves.quantilevar(fd, data, 1, 0.99) ≈ 3.573835142617284 rtol=.01

        end

        @testset "quantilecint(::DependentScalingModel)" begin

            @test_throws AssertionError IDFCurves.quantilecint(fd, data, 0, 0.99, 0.95)
            @test_throws AssertionError IDFCurves.quantilecint(fd, data, 1, 1, 0.95)
            @test_throws AssertionError IDFCurves.quantilecint(fd, data, 1, 0.99, 0)

            q_cint = IDFCurves.quantilecint(fd, data, 1, 0.99, 0.05)
            @test Distributions.mean(q_cint) ≈ IDFCurves.quantile( IDFCurves.getmarginalmodel(fd), 1, 0.99)
            @test all( q_cint .≈ [41.68545615870751, 49.09591917590525] ) 
            
        end

    end

    @testset "fitting a dependent scaling model with constrained parameters" begin

        df = CSV.read(joinpath("..", "data","702S006.csv"), DataFrame)
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        abstract_model = DependentScalingModel{SimpleScaling, UncorrelatedStructure, IdentityCopula}

        d₀ = 1.0
        initialvalues = [20, 5, 0.00001, .76]
        fixedvalues = [nothing, nothing, 0.0, nothing]

        @testset "initial shape parameter adjustment" begin
            initialvalues_test = copy(initialvalues)
            fixedvalues_test = copy(fixedvalues)

            fd = IDFCurves.fit_mle(abstract_model, data, d₀, initialvalues_test, fixedvalues_test)
            
            # Check if the initial shape parameter has been adjusted
            @test initialvalues_test[3] == 0.0001
            @test fixedvalues_test[3] == 0.0001
        end

        @testset "optimization returns expected model" begin
            initialvalues_test = copy(initialvalues)
            fd1 = IDFCurves.fit_mle(abstract_model, data, d₀, initialvalues_test)
            fd2 = IDFCurves.fit_mle(abstract_model, data, d₀, initialvalues_test, [nothing, nothing, 0.04, nothing])

            @test all([params(getmarginalmodel(fd1))...] .!= [params(getmarginalmodel(fd2))...])
            @test [params(getmarginalmodel(fd2))...] ≈ [18.161000347306484, 5.299451673789118, 0.04, 0.6945575661249878] rtol = .01
        end
    end

end