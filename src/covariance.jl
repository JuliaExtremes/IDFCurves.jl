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
