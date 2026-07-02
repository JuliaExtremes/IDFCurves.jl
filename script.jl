using Pkg
pkg"activate ."

using IDFCurves, LinearAlgebra

g(u) = [u, u^2]   # example only

Ainv = Matrix{Float64}(I, 2, 2)

ρ = IDFCurves.CvMKernel(g, Ainv)

@time λ = IDFCurves.approx_eigenvalues(ρ, 100)