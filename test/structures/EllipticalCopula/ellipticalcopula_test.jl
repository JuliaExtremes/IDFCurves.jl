
@testset "EllipticalCopula" begin
    
    @testset "GaussianCopula constructor" begin
        C = GaussianCopula([1 0; 0 1])
        @test IDFCurves.getcormatrix(C) == PDMat([1 0; 0 1])
    end

    @testset "logpdf(::GaussianCopula)" begin
        C = GaussianCopula([1 .5; .5 1])
        @test logpdf(C, [.25, .2]) ≈ 0.32840717930070484 # Value from Copulas.jl
    end

    @testset "TCopula constructor" begin
        C = TCopula(5, [1 0; 0 1])
        @test dof(C) == 5
        @test IDFCurves.getcormatrix(C) == PDMat([1 0; 0 1])
    end
    
    @testset "logpdf(::TCopula)" begin
        C = TCopula(5, [1 .5; .5 1])
        @test logpdf(C, [.25, .2]) ≈0.409866913801614 # Value from Copulas.jl
    end

end