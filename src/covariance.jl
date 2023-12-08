"""
    matern(d::Real, ν::Real, ρ::Real)

Return the Matérn correlation of parameter (ν, ρ) between measurements taken at two points separated by the distance `d`.

### Details
[Matérn covariance function](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
"""
function matern(d::Real, ν::Real, ρ::Real)
    @assert d ≥ 0 "distance must be non-negative."
    @assert ν > 0 "ν must be positive."
    @assert ρ > 0 "ρ must be positive."

    if d ≈ 0.
        lc = 0.
    else
        lc = (1-ν)*log(2) - SpecialFunctions.loggamma(ν) + ν*log(sqrt(2*ν)*d/ρ)+log(SpecialFunctions.besselk(ν,sqrt(2*ν)*d/ρ))
    end

    return exp(lc)

end

"""
    logdist(x₁::Real, x₂::Real)

Logarithmic distance between the two positive points `x₁` and `x₂`.

### Details

The logarihmic distance between `x₁ > 0` and `x₂ > 0` is defined as follows:

``h(x₁,x₂) = | \\log x₁ - \\log x₂ |.``

"""
function logdist(x₁::Real, x₂::Real)
    @assert x₁ > 0 "point must be positive."
    @assert x₂ > 0 "point must be positive."

    return abs(log(x₁) - log(x₂))

end

"""
    logdist(x::AbstractVector{<:Real})

Logarithmic distances between all pairs of points in `x`.

### Details

The function returns a square symmetric matrix of the lenght of `x`.
"""
function logdist(x::AbstractVector{<:Real})

    T = Matrix{Float64}(undef, length(x), length(x))

    for i in eachindex(x)
        for j in eachindex(x)
            (i ≤ j) ? T[i,j] = logdist(x[i], x[j]) : continue
        end
    end

    return Symmetric(T)

end