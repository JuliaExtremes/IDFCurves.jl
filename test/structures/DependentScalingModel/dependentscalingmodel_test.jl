
@testset "DependentScalingModel construction" begin
    
    d = [0. 1.; 1. 0.]
    pd = dGEV(1, 1, 1, 0, .8, .5)
    Σ = MaternCorrelationStructure(10., 1.)
    C = GaussianCopula

    dm = DependentScalingModel(pd, Σ, C)

    @test getmarginalmodel(dm) == pd
    @test getcopulatype(dm) == C
    @test getcorrelogram(dm) == Σ

end

@testset "get type of DependentScalingModel" begin
    obj = DependentScalingModel{dGEV, MaternCorrelationStructure, GaussianCopula}

    @test IDFCurves.getmarginaltype(obj) == dGEV
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

    mm = dGEV(1, 0, 1, 0, .8, .5)

    Σ = MaternCorrelationStructure(1., 1.)
    h = IDFCurves.logdist(durations)
    C = GaussianCopula(cor.(Σ, h))

    pd = DependentScalingModel(mm, Σ, GaussianCopula)

    # TODO: Verify this value
    @test loglikelihood(pd, data) ≈ -7.090619315218428

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