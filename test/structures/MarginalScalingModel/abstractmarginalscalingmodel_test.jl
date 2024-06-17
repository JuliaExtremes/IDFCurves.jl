@testset "MarginalScalingModel" begin

    include("simplescaling_test.jl")
    include("generalscaling_test.jl")

    @testset "scalingtype" begin
        ss = SimpleScaling(1., 20., 5., 0.05, 0.76)
        @test IDFCurves.scalingtype(ss) == SimpleScaling
        dGEVmodel = GeneralScaling(1., 20., 5., 0.05, 0.76, 0.07)
        @test IDFCurves.scalingtype(dGEVmodel) == GeneralScaling
    end

    @testset "godambe()" begin
        df = IDFCurves.dataset("702S006")
        tags = names(df)[2:10]
        durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
        duration_dict = Dict(zip(tags, durations))
        data = IDFdata(df, "Year", duration_dict)

        dGEVmodel = GeneralScaling(1., 19.79111996970814, 5.593814508858256, 0.040488262824997547, 0.760941914597227, 0.06814281866199222)

        G = IDFCurves.godambe(dGEVmodel, data)

        @test G ≈ [6.87679    -4.73174   -0.809215   -133.927     -59.4472
            -4.73174    10.8232    24.9278      106.388     -45.7429
            -0.809215   24.9278   949.313        40.0327      4.68722
            -133.927     106.388     40.0327    12163.9     -6921.63
            -59.4472    -45.7429     4.68722   -6921.63    13422.0] rtol=.01
    end

end