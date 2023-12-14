
@testset "DependentScalingModel construction" begin
    
    pd = dGEV(1, 1, 1, 0, .8, .5)
    C = TCopula(15, [1. .5; .5 1])

    dm = DependentScalingModel(pd, C)

    @test getmarginalmodel(dm) == pd
    @test getcopula(dm) == C

end

@testset "get type of DependentScalingModel" begin
    obj = DependentScalingModel{dGEV, GaussianCopula}

    @test IDFCurves.getmarginaltype(obj) == dGEV
    @test IDFCurves.getcopulatype(obj) == GaussianCopula

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

    mm = dGEV(1, 1, 1, 0, .8, .5)
    C = TCopula(15, [1. .5; .5 1])
    pd = DependentScalingModel(mm, C)

    @test loglikelihood(pd, data) ≈ -6.330260155320674

end

@testset "fit_mle(::DependenceScalingModel)" begin
    #TODO
end

@testset "hessian(::DependenceScalingModel)" begin
    #TODO
end

@testset "quantilevar(::DependenceScalingModel)" begin
    #TODO
end

@testset "quantilecint(::DependenceScalingModel)" begin
    #TODO
end