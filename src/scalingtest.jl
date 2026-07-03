"""
    scalingtest(pd_type::Type{<:MarginalScalingModel}, data::IDFdata;
        tag_out = nothing, q::Integer = 100)

Perform the training-validation Cramér--von Mises goodness-of-fit test for a
scaling model.

The duration identified by `tag_out` is used as the validation duration. If
`tag_out` is not provided, the smallest observed duration is used.

The argument `q` is the number of quadrature points used to approximate the
eigenvalues of the covariance kernel for the Zolotarev approximation.
"""
function scalingtest(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata;
    tag_out = nothing,
    q::Integer = 100,
)

    q > 0 || throw(ArgumentError("q must be positive."))

    tags = gettag(data)

    if tag_out === nothing
        durations = [getduration(data, tag) for tag in tags]
        tag_out = tags[argmin(durations)]
    elseif !(tag_out in tags)
        throw(ArgumentError(
            "duration tag $tag_out does not correspond to an observed duration.",
        ))
    end

    # Validation duration and validation data
    d_out = getduration(data, tag_out)
    y = getdata(data, tag_out)
    ℓ = length(y)

    ℓ > 0 || throw(ArgumentError(
        "The validation sample must contain at least one observation.",
    ))

    # Training data
    train_data = excludeduration(data, d_out)

    # Fit the scaling model using the training durations only
    fitted_model = fit_mle(pd_type, train_data, d_out)

    # Test statistic
    Fθ̂ = getdistribution(fitted_model, d_out)
    S = cvmcriterion(Fθ̂, y)

    # Observed information matrix.
    #
    # If H is the observed information summed over the training sample, then
    # H / ℓ corresponds to
    #
    #     (m / ℓ) * Î_m = a * Î_m,
    #
    # which is the matrix entering the covariance kernel.
    H = hessian(fitted_model, train_data)
    A = Symmetric(H / ℓ)

    # Covariance kernel
    g = get_g(fitted_model, d_out)
    ρ = cvmkernel(g, A)

    # Eigenvalues of the covariance kernel
    λs = approx_eigenvalues(ρ, q)

    # Zolotarev approximation of the CDF in the upper tail
    cdf_approx = zolotarev_approx(λs, S)

    # Return the p-value
    return 1 - cdf_approx
end

"""
    get_g(fd::MarginalScalingModel, d::Real)

Return the function `g` involved in the covariance kernel of the
training-validation Cramér--von Mises statistic.

For `0 < u < 1`, `g(u)` returns the gradient, with respect to the model
parameters, of the CDF of the marginal distribution at duration `d`, evaluated at

    x = F̂_d^{-1}(u),

where `F̂_d` is the fitted marginal distribution at duration `d`.
"""
function get_g(fd::MarginalScalingModel, d::Real)

    pd_type = typeof(fd)
    d₀ = duration(fd)
    θ̂ = collect(params(fd))

    # Fitted marginal distribution at the validation duration.
    pd = getdistribution(fd, d)

    function g(u::Real)
        0 < u < 1 || throw(ArgumentError("g is only defined for 0 < u < 1."))

        x = quantile(pd, u)

        function F(θ::AbstractVector{<:Real})
            return cdf(
                construct_model(
                    pd_type,
                    d₀,
                    map_to_real_space(pd_type, θ),
                ),
                d,
                x,
            )
        end

        return ForwardDiff.gradient(F, θ̂)
    end

    return g
end


"""
    cvmcriterion(pd::UnivariateDistribution, x::AbstractVector{<:Real})

Compute the Cramér--von Mises statistic between the distribution `pd` and the data vector `x`.

# Details

The statistic is

    1/(12n) + sum((F(x_(i)) - (2i - 1)/(2n))^2, i = 1:n),

where `x_(i)` denotes the ordered sample.
"""
function cvmcriterion(pd::UnivariateDistribution, x::Vector{<:Real})
    n = length(x)
    n > 0 || throw(ArgumentError("x must contain at least one observation."))

    x̃ = sort(x)

    ω² = 1/(12*n) + sum( ((2*i-1)/(2*n) - cdf(pd,x̃[i]) )^2 for i=1:n)

    return ω²

end

"""
    cvmkernel(g, A)

Return the covariance kernel used in the limiting distribution of the
training-validation Cramér--von Mises statistic.

The returned function is

    ρ(u, v) = min(u, v) - u*v + g(u)' * A^{-1} * g(v),

where `A` is typically `a * Î_m`. Equivalently, if `H` is the observed
information matrix summed over the training sample and `ℓ` is the validation
sample size, one may use `A = H / ℓ`.

The matrix factorization of `A` is computed once and reused each time the kernel
is evaluated.
"""
function cvmkernel(g, A::AbstractMatrix)
    Afact = factorize(A)

    function ρ(u::Real, v::Real)
        gu = g(u)
        gv = g(v)

        return min(u, v) - u * v + dot(gu, Afact \ gv)
    end

    return ρ
end


"""
    approx_eigenvalues(ρ, q::Integer)

Approximate the eigenvalues of the integral operator with kernel `ρ(u, v)` on
`[0, 1]`, using a midpoint Nyström approximation with `q` quadrature points.

The returned eigenvalues are sorted in decreasing order.
"""
function approx_eigenvalues(ρ::K, q::Integer) where {K}
    q > 0 || throw(ArgumentError("q must be positive."))

    Kmat = Matrix{Float64}(undef, q, q)

    for j in 1:q
        v = (2j - 1) / (2q)

        for i in 1:j
            u = (2i - 1) / (2q)
            Kmat[i, j] = ρ(u, v) / q
        end
    end

    λ = eigvals(Symmetric(Kmat, :U))

    return reverse(λ)
end


"""
    zolotarev_approx(λs::AbstractVector{<:Real}, x::Real; tail_threshold = 0.95)

Return a Zolotarev upper-tail approximation of the CDF of the sum of λᵢ Zᵢ² where the `Zᵢ` are independent standard normal random variables.

The approximation is intended for large values of `x`, corresponding to CDF values close to one.
"""
function zolotarev_approx(
    λs::AbstractVector{<:Real},
    x::Real;
    tail_threshold::Real = 0.95,
    atol::Real = 1e-12,
    rtol::Real = 1e-10,
)
    x > 0 || throw(ArgumentError("x must be positive."))

    # Keep only positive eigenvalues and sort them in decreasing order.
    λ = sort(filter(λᵢ -> λᵢ > 0, Float64.(λs)); rev = true)

    length(λ) >= 1 || throw(ArgumentError(
        "The vector of eigenvalues must contain at least one positive element.",
    ))

    γ₁ = λ[1]

    # Multiplicity of the largest eigenvalue, up to numerical tolerance.
    m₁ = count(λᵢ -> isapprox(λᵢ, γ₁; atol = atol, rtol = rtol), λ)

    # Eigenvalues strictly smaller than the largest one.
    λrest = λ[(m₁ + 1):end]

    log_product_term =
        isempty(λrest) ? 0.0 :
        -sum(0.5 * log1p(-λᵢ / γ₁) for λᵢ in λrest)

    log_tail =
        log_product_term -
        loggamma(0.5 * m₁) +
        (0.5 * m₁ - 1) * log(x / (2γ₁)) -
        x / (2γ₁)

    approx_cdf = 1 - exp(log_tail)

    approx_cdf = clamp(approx_cdf, 0.0, 1.0)

    if approx_cdf < tail_threshold
        @warn "Zolotarev approximation is outside its recommended upper-tail domain." approx_cdf tail_threshold
    end

    return approx_cdf
end