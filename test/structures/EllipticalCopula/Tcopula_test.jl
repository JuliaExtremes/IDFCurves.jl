@testset "TCopula" begin
    
    @testset "TCopula constructor" begin
        @test_throws AssertionError C = TCopula{5.1}([1 0; 0 1])

        C = TCopula{5}([1 0; 0 1])
        @test dof(C) == 5
        @test IDFCurves.getcormatrix(C) == PDMat([1 0; 0 1])

        C = TCopula(6, [1 .5; .5 1])
        @test dof(C) == 6
        @test IDFCurves.getcormatrix(C) == PDMat([1 .5; .5 1])

        C_type = TCopula{7}
        @test dof(C_type)  == 7

    end
    
    @testset "logpdf(::TCopula)" begin
        C = TCopula{5}([1 .5; .5 1])
        @test logpdf(C, [.25, .2]) ≈ 0.409866913801614 # Value from Copulas.jl
    end

end
