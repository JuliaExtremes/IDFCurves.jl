@testset "IdentityCopula" begin
    
    @testset "IdentityCopula constructor" begin
        C1 = IdentityCopula()
        C2 = IdentityCopula([1 0; 0 1])
        @test typeof(C1) == IdentityCopula
        @test C1 == C2
    end

    @testset "logpdf(::IdentityCopula)" begin
        C = IdentityCopula()
        @test logpdf(C, [.5]) ≈ 0.
        @test logpdf(C, [.25, .2]) ≈ 0.
    end

end