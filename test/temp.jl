using Pkg
Pkg.activate(".")

using Distributions, Test

using IDFCurves

import IDFCurves.largescale_offset

"""
    scaling_factor(sm::UniversalScaling, d::Real)

Compute the scaling factor for duration `d` under the Universal Scaling model `sm`.

### Details

The scaling factor is defined as

```math
s(d) = \frac{(d+\\delta)^{-\\alpha}+\\tau}{(d_0+\\delta)^{-\\alpha}+\\tau}.
```
"""
function scaling_factor(sm::UniversalScaling, d::Real)
    @assert d>0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    α = exponent(sm)
    δ = offset(sm)
    τ = largescale_offset(sm)

    s = (exp(-α * log(d + δ)) + τ) / (exp(-α * log(d₀ + δ)) + τ)

    return s

end

@testset "scaling_factor(sm::UniversalScaling, d::Real)" begin
    import IDFCurves.scaling_factor

    d₀, μ₀, σ₀, ξ, α, δ, τ = (3, 1., 1., 0., .5, 1., .5)
    pd = UniversalScaling(d₀, μ₀, σ₀, ξ, α, δ, τ)

    @test_throws AssertionError scaling_factor(pd, -1) 
    
    # No scaling
    @test scaling_factor(pd, d₀) ≈ 1.

    #Scaling
    @test scaling_factor(pd, 15.) ≈ .75
end


"""
    scaling_factor(sm::GeneralScaling, d::Real)

Compute the scaling factor for duration `d` under the General Scaling model `sm`.

### Details

The scaling factor is defined as

```math
s(d) = \frac{(d+\\delta)^{-\\alpha}}{(d_0+\\delta)^{-\\alpha}}.
```
"""
function scaling_factor(sm::GeneralScaling, d::Real)

    @assert d>0 "Duration should be positive, got d = $d."

    d₀ = duration(sm)
    α = exponent(sm)
    δ = offset(sm)

    s = exp(-α * log1p((d - d₀) / (d₀ + δ)))

    return s

end

@testset "scaling_factor(sm::GeneralScaling, d::Real)" begin
    import IDFCurves.scaling_factor

    d₀, μ₀, σ₀, ξ, α, δ = (.5, 1., 1., 0., .5, .5)
    pd = GeneralScaling(d₀, μ₀, σ₀, ξ, α, δ)

    @test_throws AssertionError scaling_factor(pd, -1) 
    
    # No scaling
    @test scaling_factor(pd, d₀) ≈ 1.

    #Scaling
    @test scaling_factor(pd, 3.5) ≈ .5
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