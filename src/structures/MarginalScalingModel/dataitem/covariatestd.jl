
struct CovariateStd <: DataItem
    name :: String
    value :: Vector{<:Real}
    offset :: Real
    scale :: Real
end

"""
    CovariateStd(name::String, z::Vector{<:Real})::CovariateStd

Construct a CovariateStd type from the standardized vector `z` with the name `name`.

"""
function CovariateStd(name::String, z::Vector{<:Real})::CovariateStd

    m = mean(z)
    s = std(z)

    @assert isapprox(m, 0.0, atol = sqrt(eps())) "the mean should be equal to zero. Use the type Covariate instead."
    @assert isapprox(s, 1.0, atol = sqrt(eps())) "the standard deviation should be equal to one. Use the type Covariate instead."

    return CovariateStd(name, z, 0, 1)
end




"""
    reconstruct(vstd::CovariateStd)

Reconstruct the original Covariate from the standardized one.

See also

See also [`Covariate`](@ref), [`CovariateStd`](@ref)  and [`standardize`](@ref).
"""
function reconstruct(vstd::CovariateStd)::Covariate

    z = vstd.value

    x = vstd.scale*z .+ vstd.offset

    v = Covariate(vstd.name, x)

    return v

end
