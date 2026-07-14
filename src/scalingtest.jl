
struct CvMValidationTest{
    M,
    S<:Real,
    D<:ContinuousUnivariateDistribution,
}
    fitted_model::M
    test_statistic::S
    null_distribution::D
end

function Base.show(io::IO, test_struct::CvMValidationTest)
    println(io, "CvMValidationTest(")
    println(io, "  test_statistic = ", test_struct.test_statistic)
    println(io, "  null_distribution = ", typeof(test_struct.null_distribution))
    println(io, "  fitted_model = ", typeof(test_struct.fitted_model))
    print(io, ")")
end


"""
    pvalue(test)

Return the upper-tail p-value of the validation statistic.
"""
function pvalue(test_struct::CvMValidationTest)
    return ccdf(test_struct.null_distribution, test_struct.test_statistic)
end

"""
    decision_threshold(test, α = 0.05)

Return the rejection threshold at level `α`.
"""
function decision_threshold(test_struct::CvMValidationTest, α::Real=0.05)
    0 < α < 1 || throw(ArgumentError("The test level should be in (0, 1), got $α."))

    return quantile(test_struct.null_distribution, 1. - α)
end

"""
    decision(test, α = 0.05)

Return whether the validation test rejects the null hypothesis at level `α`.
"""
function decision(test_struct::CvMValidationTest, α::Real=0.05)
    threshold = decision_threshold(test_struct, α)

    return test_struct.test_statistic > threshold
end






"""
    scalingtest(pd_type::Type{<:MarginalScalingModel}, data::IDFdata;
        tag_out = nothing, q::Integer = 20, nquad::Integer = max(5q, q + 20))

Perform the training-validation Cramér--von Mises goodness-of-fit test for a
scaling model.

The duration identified by `tag_out` is used as the validation duration. If
`tag_out` is not provided, the smallest observed duration is used.

The argument `q` is the number of retained eigenvalues used in the finite
approximation of the Cramér--von Mises null distribution.
"""
function scalingtest(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata;
    tag_out = nothing,
    q::Integer = 20,
)
    q > 0 || throw(ArgumentError("q must be positive."))

    if tag_out === nothing
        tag_out = _validation_tag(data)
    else
        tag_out = _validation_tag(data, tag_out)
    end

    # Validation duration and validation data
    d_out = getduration(data, tag_out)
    y = getdata(data, tag_out)
    ℓ = length(y)

    ℓ > 0 || throw(ArgumentError(
        "The validation sample must contain at least one observation.",
    ))

    # Training data
    train_data = excludeduration(data, tag_out)

    # Fit the scaling model using the training durations only
    fitted_model = fit_mle(pd_type, train_data, d_out)

    # Test statistic
    F = getdistribution(fitted_model, d_out)
    S = cvmcriterion(F, y)

    # Observed information matrix.
    #
    # If H is the observed information summed over the training sample, then
    # H / ℓ corresponds to the scaled information matrix entering the
    # covariance kernel.
    H = hessian(fitted_model, train_data)
    A = Symmetric(H / ℓ)

    # Covariance kernel
    g = get_g(fitted_model, d_out)
    ρ = cvmkernel(g, A)

    # Eigenvalues of the covariance kernel
    λ = approx_eigenvalues(ρ, q)

    # Test statistic distribution
    pd = CvMDistribution(λ)

    return CvMValidationTest(fitted_model, S, pd)
end

# Covariance kernel

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
    approx_eigenvalues(ρ, q; nquad = max(5q, q + 20), eigentol = sqrt(eps(Float64)))

Approximate the largest `q` eigenvalues of the integral operator with kernel
`ρ(u, v)` on `[0, 1]`, using a midpoint Nyström approximation with `nquad`
quadrature points.

### Details

The returned eigenvalues are sorted in decreasing order. The ignored tail is
not approximated.

The method is a quadrature-based Nyström approximation for the eigenvalues of
a compact integral operator; see Atkinson (1975).

#### Reference

Atkinson, K. E. (1975). Convergence rates for approximate eigenvalues of compact integral operators. *SIAM Journal on Numerical Analysis, 12(2), 213–222*. https://doi.org/10.1137/0712020
"""
function approx_eigenvalues(
    ρ::K,
    q::Integer;
    nquad::Integer=max(5q, q + 20),
    eigentol::Real=sqrt(eps(Float64))) where {K}

    q > 0 || throw(ArgumentError("q must be positive."))
    nquad >= q || throw(ArgumentError("nquad must be at least q."))
    eigentol >= 0 || throw(ArgumentError("eigentol must be non-negative."))
    isfinite(eigentol) || throw(ArgumentError("eigentol must be finite."))

    Kmat = Matrix{Float64}(undef, nquad, nquad)

    for j in 1:nquad
        v = (2j - 1) / (2nquad)

        for i in 1:j
            u = (2i - 1) / (2nquad)
            Kmat[i, j] = ρ(u, v) / nquad
        end
    end

    λraw = eigvals(Symmetric(Kmat, :U))

    λmax = maximum(abs, λraw)
    scale = max(λmax, 1.0)

    if any(λ -> λ < -eigentol * scale, λraw)
        throw(ArgumentError("Negative eigenvalue beyond numerical tolerance."))
    end

    λ = sort([λ for λ in λraw if λ > eigentol * scale]; rev=true)

    length(λ) >= q || throw(ArgumentError("Fewer than q positive eigenvalues were found."))

    return λ[1:q]
end


# Cramér-von Mises statistic

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

    ω² = 1/(12*n) + sum(((2*i-1)/(2*n) - cdf(pd, x̃[i]))^2 for i=1:n)

    return ω²

end

# Computing p-values

"""
    zolotarev_approx(λs::AbstractVector{<:Real}, x::Real; tail_threshold = 0.95)

Return a Zolotarev upper-tail approximation of the CDF of the sum of λᵢ Zᵢ² where the `Zᵢ` are independent standard normal random variables.

The approximation is intended for large values of `x`, corresponding to CDF values close to one.

Note: we do not use this approximation anymore and now rely on a method similar to GeneralizedChisqDistribution.jl to give p-values over the whole range.
"""
function zolotarev_approx(
    λs::AbstractVector{<:Real},
    x::Real;
    tail_threshold::Real=0.95,
    atol::Real=1e-12,
    rtol::Real=1e-10,
)
    x > 0 || throw(ArgumentError("x must be positive."))

    # Keep only positive eigenvalues and sort them in decreasing order.
    λ = sort(filter(λᵢ -> λᵢ > 0, Float64.(λs)); rev=true)

    length(λ) >= 1 || throw(ArgumentError(
        "The vector of eigenvalues must contain at least one positive element.",
    ))

    γ₁ = λ[1]

    # Multiplicity of the largest eigenvalue, up to numerical tolerance.
    m₁ = count(λᵢ -> isapprox(λᵢ, γ₁; atol=atol, rtol=rtol), λ)

    # Eigenvalues strictly smaller than the largest one.
    λrest = λ[(m₁+1):end]

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





"""
    scalingtest_bootstrap(fitted_model::MarginalScalingModel, data::IDFdata;
        tag_out = nothing, B::Integer = 999, rng = Random.default_rng())

Generate dependence-aware bootstrap replicates of the training-validation
Cramér--von Mises goodness-of-fit test.

The bootstrap resamples entire yearly rank vectors across durations. This
preserves the empirical cross-duration dependence while imposing the fitted
scaling model on the marginal distributions through `fitted_model`.

The duration identified by `tag_out` is used as the validation duration. If
`tag_out` is not provided, the smallest observed duration is used.

The function returns a vector of `CvMValidationTest` objects, one for each
bootstrap sample.
"""
function scalingtest_bootstrap(
    fitted_model::MarginalScalingModel,
    data::IDFdata;
    tag_out = nothing,
    B::Integer = 999,
    rng = Random.default_rng(),
)
    B > 0 || throw(ArgumentError("B must be positive."))

    pd_type = scalingtype(fitted_model)

    tags = gettag(data)
    if tag_out === nothing
        tag_out = _validation_tag(data)
    else
        tag_out = _validation_tag(data, tag_out)
    end

    # Restrict the bootstrap to complete years across all durations.
    years = _common_years(data, tags)

    length(years) > 0 || throw(ArgumentError(
        "There is no common year across all durations.",
    ))

    data_common = _restrict_years(data, years; tags = tags)

    # Empirical cross-duration dependence, represented by yearly rank vectors.
    U = _pseudoobs_matrix(data_common, tags)
    n = size(U, 1)

    Tstar = Vector{CvMValidationTest}(undef, B)

    # Generate bootstrap indices sequentially to avoid sharing the RNG across threads.
    bootstrap_indices = [rand(rng, 1:n, n) for _ in 1:B]

    for b in 1:B
        idx = bootstrap_indices[b]
        Ustar = U[idx, :]

        data_star = _idfdata_from_pseudoobs(data_common, fitted_model, Ustar)

        Tstar[b] = scalingtest(pd_type, data_star; tag_out = tag_out)
    end

    return Tstar
end


"""
    _pseudoobs_matrix(data::IDFdata, tags::AbstractVector{<:AbstractString})

Return the matrix of rank-based pseudo-observations for the durations identified by `tags`.

### Details

Each column corresponds to one duration and is obtained as `tiedrank(y) / (n + 1)`.
All durations must have the same number of common years.
"""
function _pseudoobs_matrix(
    data::IDFdata,
    tags::AbstractVector{<:AbstractString},
)

    n = length(getdata(data, tags[1]))
    p = length(tags)

    U = Matrix{Float64}(undef, n, p)

    for (j, tag) in enumerate(tags)
        y = getdata(data, tag)

        length(y) == n || throw(ArgumentError(
            "All durations must have the same number of common years.",
        ))

        # Rank transformation
        U[:, j] .= StatsBase.tiedrank(y) ./ (n + 1)
    end

    return U
end

"""
    _idfdata_from_pseudoobs(template, fitted_model, U)

Transform pseudo-observations into an `IDFdata` object using the fitted marginal
scaling model.

Each column of `U` corresponds to one duration in `template`. For duration `d`,
pseudo-observations are mapped back to the data scale with

    quantile(getdistribution(fitted_model, d), U[i, j]).

The returned `IDFdata` keeps the duration tags and durations from `template`,
uses bootstrap years `1:n`, and contains the transformed observations.
"""
function _idfdata_from_pseudoobs(
    template::IDFdata,
    fitted_model::MarginalScalingModel,
    U::AbstractMatrix{<:Real},
)
    n, p = size(U)

    tags = gettag(template)
    
    p == length(tags) || throw(ArgumentError(
        "The number of columns in U must match the number of duration tags in template.",
    ))

    new_tag = String.(collect(tags))

    new_duration = Dict{String,Float64}()
    new_year = Dict{String,Vector{Int64}}()
    new_data = Dict{String,Vector{Float64}}()

    bootstrap_years = collect(Int64, 1:n)

    for (j, tag) in enumerate(new_tag)
        d = getduration(template, tag)
        pd = getdistribution(fitted_model, d)

        new_duration[tag] = Float64(d)
        new_year[tag] = copy(bootstrap_years)
        new_data[tag] = [Float64(quantile(pd, U[i, j])) for i in 1:n]
    end

    return IDFdata(new_tag, new_duration, new_year, new_data)
end

