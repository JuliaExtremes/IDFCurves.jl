
@testset "logpdf_TCopula" begin
    u = [.2, .7]
    ν = 5
    Σ = [1 .5; .5 1]

    C = MvTDist(ν, Σ)

    @test_throws AssertionError IDFCurves.logpdf_TCopula(C, [-0.5, 0.7])

    @test IDFCurves.logpdf_TCopula(C, u) ≈ -0.3970179543005301 # Value computed with Copulas.jl
end

@testset "matern" begin
    @test_throws AssertionError IDFCurves.matern(-1,1,1)
    @test_throws AssertionError IDFCurves.matern(1,-1,1)
    @test_throws AssertionError IDFCurves.matern(1,1,-1)

    @test IDFCurves.matern(0,1,1) ≈ 1
    @test IDFCurves.matern(2,1/2,1) ≈ exp(-2/1)
    @test IDFCurves.matern(2,3/2,1) ≈ (1+sqrt(3)*2)*exp(-sqrt(3)*2)
end

@testset "logdist(x₁::Real, x₂::Real)" begin
    @test_throws AssertionError IDFCurves.logdist(-1,1)
    @test_throws AssertionError IDFCurves.logdist(1,-1)

    @test IDFCurves.logdist(1,1) ≈ 0
    @test IDFCurves.logdist(exp(1),exp(2)) ≈ 1
end

@testset "logdist(x::AbstractVector{<:Real})" begin
    x = exp.(1:2)
    @test IDFCurves.logdist(x) ≈ [0. 1.; 1. 0.]
end

@testset "quantile_TDist2" begin
    @test_throws AssertionError IDFCurves.quantile_TDist2(TDist(3), .5)
    @test_throws AssertionError IDFCurves.quantile_TDist2(TDist(2), 1.5)

    @test IDFCurves.quantile_TDist2(TDist(2), .5) ≈ 0.
    @test IDFCurves.quantile_TDist2(TDist(2), .95) ≈ 2.919985580353724
end

@testset "quantile_TDist4" begin
    @test_throws AssertionError IDFCurves.quantile_TDist4(TDist(3), .5)
    @test_throws AssertionError IDFCurves.quantile_TDist4(TDist(4), 1.5)

    @test IDFCurves.quantile_TDist4(TDist(4), .5)  ≈ 0.
    @test IDFCurves.quantile_TDist4(TDist(4), .95) ≈ 2.1318467863266495
end

@testset "quantile_TDist" begin
    @test_throws AssertionError IDFCurves.quantile_TDist4(TDist(3), 1.5)

    @test IDFCurves.quantile_TDist(TDist(5), .3) ≈ -0.5594296444693607
    @test IDFCurves.quantile_TDist(TDist(5), .5) ≈ 0. 
    @test IDFCurves.quantile_TDist(TDist(5), .7) ≈ 0.5594296444693607

end

@testset "quantile_TDist_rtail" begin
    @test_throws AssertionError IDFCurves.quantile_TDist_rtail(TDist(5), .5)
    @test IDFCurves.quantile_TDist_rtail(TDist(5), .95) ≈ 2.015048373333023
end

@testset "quantile_TDist_ltail" begin
    @test_throws AssertionError IDFCurves.quantile_TDist_ltail(TDist(5), .5)
    @test IDFCurves.quantile_TDist_ltail(TDist(5), .05) ≈ -2.0150483733330233
end


@testset "quantile(::TDist)" begin
    @test_throws AssertionError quantile(TDist(5), 1.5)

    @test quantile(TDist(2), .95) ≈ 2.919985580353724
    @test quantile(TDist(4), .95) ≈ 2.1318467863266495

    @test quantile(TDist(5), .05) ≈ -2.0150483733330233
    @test quantile(TDist(5), .6) ≈ 0.26718086570414507
    @test quantile(TDist(5), .95) ≈ 2.015048373333023
end
