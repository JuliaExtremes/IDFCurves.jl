"""
    CvMDistribution(eigenvalues; eigentol = sqrt(eps(Float64)))

Finite-eigenvalue approximation to a Cramér--von Mises limiting statistic,

    Q = ∑ᵢ λᵢ χ²ᵢ,

where the `λᵢ` are positive eigenvalues and the `χ²ᵢ` are independent
chi-square variables with one degree of freedom.

This is the positive, central, one-degree-of-freedom special case of a
generalized chi-square distribution. It is analogous to
`GeneralizedChisq(eigenvalues, 1.0, 0.0, 0.0, 0.0)` from
GeneralizedChisqDistribution.jl, but uses a lighter representation and
methods specialized to this case.
"""
struct CvMDistribution{T<:AbstractFloat} <: ContinuousUnivariateDistribution
    eigenvalues::Vector{T}
    CvMDistribution{T}(eigenvalues::AbstractVector{T}) where {T<:Real} = new{T}(eigenvalues)
end

function CvMDistribution(eigenvalues::AbstractVector{<:Real})

    tol = sqrt(eps())

    λraw = Float64.(eigenvalues)
    isempty(λraw) && throw(ArgumentError("At least one non-zero eigenvalue is required."))

    any(≤(0), λraw) && throw(ArgumentError("All eigenvalues must be positive."))
    any(!isfinite, λraw) && throw(ArgumentError("All eigenvalues must be finite."))
    isempty(λraw) && throw(ArgumentError("At least one non-zero eigenvalue is required."))

    λ = sort(λraw; rev = true)

    return CvMDistribution{Float64}(λ)
end

"""
    params(d::CvMDistribution)

Return the eigenvalues of `d` as the distribution parameter tuple.
"""
params(d::CvMDistribution) = (d.eigenvalues,)

"""
    length(d::CvMDistribution)

Return the number of retained eigenvalues.
"""
Base.length(d::CvMDistribution) = length(d.eigenvalues)

"""
    minimum(d::CvMDistribution)

Return the lower endpoint of the support.
"""
Base.minimum(::CvMDistribution) = 0.0

"""
    maximum(d::CvMDistribution)

Return the upper endpoint of the support.
"""
Base.maximum(::CvMDistribution) = Inf

"""
    insupport(d::CvMDistribution, x)

Return whether `x` belongs to the support of `d`.
"""
insupport(::CvMDistribution, x::Real) = x >= 0

"""
    mean(d::CvMDistribution)

Return the mean `d``.
"""
mean(d::CvMDistribution) = sum(d.eigenvalues)


"""
    var(d::CvMDistribution)

Return the variance of `d`.
"""
var(d::CvMDistribution) = 2sum(abs2, d.eigenvalues)

"""
    std(d::CvMDistribution)

Return the standard deviation of `d`.
"""
std(d::CvMDistribution) = sqrt(var(d))

"""
    ccdf(d, x; rtol = 1e-10, atol = 1e-12)

Compute `P(X > x)` by numerical inversion of the characteristic function.

### Details

Similar to GeneralizedChisqDistribution.jl, because that package’s author describes its cdf as integration with QuadGK.jl.
"""
function ccdf(
    d::CvMDistribution,
    x::Real;
    rtol::Real = 1e-10,
    atol::Real = 1e-12,
)
    isnan(x) && return NaN
    x <= 0 && return 1.0
    isinf(x) && return x > 0 ? 0.0 : 1.0

    λs = d.eigenvalues
    μ = mean(d)

    function integrand(u)
        if iszero(u)
            return (μ - x) / 2
        end

        θ = -x * u / 2
        logρ = 0.0

        for λ in λs
            z = λ * u
            θ += atan(z) / 2
            logρ += log1p(abs2(z)) / 4
        end

        return sin(θ) / (u * exp(logρ))
    end

    integral, _ = QuadGK.quadgk(
        integrand,
        0.0,
        Inf;
        rtol = rtol,
        atol = atol,
    )

    return clamp(0.5 + integral / π, 0.0, 1.0)
end

"""
    quantile(d, p; xatol = 1e-10, maxiter = 200, rtol = 1e-10, atol = 1e-12)

Compute the `p`-quantile by numerically inverting `ccdf`.
"""
function quantile(
    d::CvMDistribution,
    p::Real;
    xatol::Real = 1e-10,
    maxiter::Integer = 200,
    rtol::Real = 1e-10,
    atol::Real = 1e-12,
)
    0 <= p <= 1 || throw(ArgumentError("p must be between 0 and 1."))

    p == 0 && return minimum(d)
    p == 1 && return maximum(d)

    q = 1 - p

    lower = 0.0
    upper = mean(d) + 8std(d)

    while ccdf(d, upper; rtol = rtol, atol = atol) > q
        upper *= 2
        isfinite(upper) || throw(ArgumentError("Could not bracket the quantile."))
    end

    objective(x::Real) = abs2(ccdf(d, x; rtol = rtol, atol = atol) - q)

    res = Optim.optimize(
        objective,
        lower,
        upper,
        Optim.Brent();
        abs_tol = xatol,
        iterations = maxiter,
    )

    Optim.converged(res) || throw(ArgumentError("The quantile was not found."))

    return Optim.minimizer(res)
end