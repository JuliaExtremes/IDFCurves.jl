@testset "data.jl" begin
    @testset "dataset(name)" begin
        # nonexistent file throws
        @test_throws ErrorException IDFCurves.dataset("nonexistant")

        # 702S006
        df = IDFCurves.dataset("702S006")
        @test size(df, 1) == 72
        @test size(df, 2) == 10
    end

end