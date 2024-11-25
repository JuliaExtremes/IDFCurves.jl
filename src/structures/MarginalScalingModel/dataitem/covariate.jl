"""
    Covariate(name::String, value :: Vector{<:Real})

Construct a Covariate type
"""
struct Covariate <: DataItem
    name :: String
    value :: Vector{<:Real}
end

"""
    standardize(v::Covariate)

Standardize the values of the Covariate.

# Implementation

The Covariate values are standardized by substracting the empirical mean
and dividing by the empirical standard deviation. A [`CovariateStd`](@ref) type
is returned.

See also [`Covariate`](@ref), [`CovariateStd`](@ref) and [`reconstruct`](@ref).
"""
function standardize(v::Covariate)::CovariateStd

    x = v.value

    x̄ = mean(x)
    s = std(x)

    offset = isapprox(x̄,0) ? zero(typeof(x[1])) : x̄
    scale = (isapprox(s,1) || isapprox(s,0.0)) ? one(typeof(x[1])) : s

    z = (x .- offset) ./ scale

    vstd = CovariateStd(v.name, z, offset, scale)

    return vstd

end
