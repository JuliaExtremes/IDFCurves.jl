using Pkg
Pkg.activate(".")

using Distributions, Test

using IDFCurves

@testset "loglikelihood(::UniversalScaling)" begin

        d₀ = 3
        μ₀ = 1
        σ₀ = 1
        ξ = 0
        α = .5
        δ = 1
        τ = .5
        
        pd = UniversalScaling(d₀, μ₀, σ₀, ξ, α, δ, τ)
        data = rand(pd, [1, 3], 3, tags=["1", "3"])
        y₁ = getdata(data, "1")
        y₃ = getdata(data, "3")

        scaling(d::Real) = (exp(-α * log(d + δ)) + τ) / (exp(-α * log(d₀ + δ)) + τ)
        s₁ = scaling(1)
        s₃ = scaling(3)

        ll = sum(logpdf.(GeneralizedExtremeValue(s₁*μ₀, s₁*σ₀, ξ), y₁)) + sum(logpdf.(GeneralizedExtremeValue(s₃*μ₀, s₃*σ₀, ξ), y₃))

        @test loglikelihood(pd, data) ≈ ll
    end


