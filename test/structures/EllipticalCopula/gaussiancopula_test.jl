@testset "GaussianCopula" begin
    
    @testset "GaussianCopula constructor" begin
        C = GaussianCopula([1 0; 0 1])
        @test IDFCurves.getcormatrix(C) == PDMat([1 0; 0 1])
    end

    @testset "logpdf(::GaussianCopula)" begin
        C = GaussianCopula([1 .5; .5 1])
        @test logpdf(C, [.25, .2]) ≈ 0.32840717930070484 # Value from Copulas.jl
    end

end