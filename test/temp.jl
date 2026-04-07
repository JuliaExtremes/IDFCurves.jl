using Pkg
Pkg.activate(".")

using Distributions, Test

using IDFCurves

d₀, μ₀, σ₀, ξ, α, δ, τ = (3, 1., 1., 0., .5, 1., 1.)



import IDFCurves.minimum_intensity

function scaling(sm::UniversalScaling, d::Real)
    @assert d>0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    α = exponent(sm)
    δ = offset(sm)
    τ = minimum_intensity(sm)

    s = (exp(-α * log(d + δ)) + τ) / (exp(-α * log(d₀ + δ)) + τ)

    return s

end

pd = UniversalScaling(d₀, μ₀, σ₀, ξ, α, δ, τ)
@time scaling(pd, 3)


function scaling(sm::GeneralScaling, d::Real)

    @assert d>0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    α = exponent(sm)
    δ = offset(sm)

    s = exp(-α * log1p((d - d₀) / (d₀ + δ)))

    return s

end

pd = GeneralScaling(d₀, μ₀, σ₀, ξ, α, δ)
@time scaling(pd, 1)

function scaling(sm::SimpleScaling, d::Real)

    @assert d>0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    α = exponent(sm)

    s = exp(-α * log1p((d - d₀) / d₀))

    return s

end

pd = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
@time scaling(pd, 1)


@testset "scaling()" begin

    @testset "scaling(sm::UniversalScaling, d::Real)" begin

        d₀, μ₀, σ₀, ξ, α, δ, τ = (3., 1., 1., 0., 1., .5, 1.)
        sm = UniversalScaling(d₀, μ₀, σ₀, ξ, α, δ, τ)
        
    end

end

d₀, μ₀, σ₀, ξ, δ, α, τ = (3., 1., 1., 0., .5, 1., 1.)
sm = UniversalScaling(d₀, μ₀, σ₀, ξ, δ, α, τ)

@test scaling(sm ,8.) ≈ .75

# TODO Change parameter name
# \alpha :exponent
# \delta : small-scale offset
# \tau : large-scale offset