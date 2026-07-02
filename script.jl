using Pkg
pkg"activate ."

using Distributions, IDFCurves, LinearAlgebra, Test

g(u) = [u, u^2]   # example only

Ainv = Matrix{Float64}(I, 2, 2)

ρ = IDFCurves.CvMKernel(g, Ainv)

@time λ = IDFCurves.approx_eigenvalues(ρ, 100)




λs = [1.0]
x = Distributions.quantile(Distributions.Chisq(1), 0.99)


IDFCurves.zolotarev_approx(λs, x)

@test isapprox(
        IDFCurves.zolotarev_approx(λs, x),
        0.99;
        rtol = 1e-2,
    )




    λs = Float64[]
    for k in 1:10
        append!(λs, fill(1 / 2^k, 2k^2))
    end

    IDFCurves.zolotarev_approx(λs, 16.51)

    # The value 16.51 was obtained by simulation.
    @test isapprox(
        IDFCurves.zolotarev_approx(λs, 16.51),
        0.99;
        rtol = 1e-2,
    )