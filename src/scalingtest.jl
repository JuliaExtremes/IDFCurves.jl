
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
    CvMComponents(cdf_gradient, information_factor)

Components defining the parameter-estimation correction in a Cramér--von Mises
covariance kernel.

The field `cdf_gradient` is a callable object returning the gradient of the
fitted validation CDF at a probability level, and `information_factor` is a
Cholesky factorization of the scaled information matrix used to solve systems involving
its inverse.
"""
struct CvMComponents{G,F<:Cholesky}
    cdf_gradient::G
    information_factor::F
end




"""
    scalingtest(::Type{<:MarginalScalingModel}, data::IDFdata;
        tag_out=nothing, q::Integer=20)

    scalingtest(::Type{<:MarginalScalingModel}, data::IDFdata,
        initialmodel::MarginalScalingModel;
        tag_out=nothing, q::Integer=20)

Perform the training-validation Cramér--von Mises test for a marginal scaling
model, optionally using a supplied initial model.

The duration identified by `tag_out` is used for validation. If `tag_out` is
not provided, the smallest observed duration is used. The argument `q` is the
number of eigenvalues retained to approximate the null distribution of the
test statistic.
"""
function scalingtest end


function scalingtest(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata;
    tag_out=nothing,
    q::Integer=20,
)

    q > 0 || throw(ArgumentError("q must be positive."))

    tag_out = if isnothing(tag_out)
        _validation_tag(data)
    else
        _validation_tag(data, tag_out)
    end

    train_data = excludeduration(data, tag_out)
    initialmodel = initialize(pd_type, train_data, 1.0)

    return scalingtest(
        pd_type,
        data,
        initialmodel;
        tag_out=tag_out,
        q=q,
    )
end

function scalingtest(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata,
    initialmodel::MarginalScalingModel;
    tag_out = nothing,
    q::Integer = 20,
)
    q > 0 || throw(ArgumentError("q must be positive."))

    scalingtype(initialmodel) === pd_type ||
          throw(ArgumentError("Model and initial model must be of the same type, got $pd_type ≠ $(scalingtype(initialmodel))"))

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
    fitted_model = fit_mle(pd_type, train_data, initialmodel)

    # Test statistic
    F = getdistribution(fitted_model, d_out)
    S = cvmcriterion(F, y)

    # CvM kernel components
    cvm_components = compute_cvm_components(fitted_model, train_data, d_out, ℓ)
    λ = approx_eigenvalues(cvm_components, q)

    # Test statistic distribution
    pd = CvMDistribution(λ)

    return CvMValidationTest(fitted_model, S, pd)
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

"""
    validation_cvm_statistic(
        ::Type{<:MarginalScalingModel},
        data::IDFdata;
        tag_out=nothing,
    )

    validation_cvm_statistic(
        ::Type{<:MarginalScalingModel},
        data::IDFdata,
        initialmodel::MarginalScalingModel;
        tag_out=nothing,
    )

Compute the training-validation Cramér--von Mises statistic for a marginal
scaling model, optionally using a supplied initial model.

The duration identified by `tag_out` is used for validation. If `tag_out` is
not provided, the smallest observed duration is used.

### Detail

Lightweight version of scaling test when only the test statistic matters, and not the test statistic distribution under the null hypothesis.
"""
function validation_cvm_statistic end

function validation_cvm_statistic(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata,
    initialmodel::MarginalScalingModel;
    tag_out=nothing,
)

    scalingtype(initialmodel) === pd_type ||
        throw(ArgumentError(
            "Model and initial model must be of the same type.",
        ))

    if isnothing(tag_out)
        tag_out = _validation_tag(data)
    else
        tag_out = _validation_tag(data, tag_out)
    end

    d_out = getduration(data, tag_out)
    y_out = getdata(data, tag_out)

    isempty(y_out) && throw(ArgumentError(
        "The validation sample must contain at least one observation.",
    ))

    train_data = excludeduration(data, tag_out)

    fitted_model = fit_mle(pd_type, train_data, initialmodel)

    distribution = getdistribution(fitted_model, d_out)
    test_statistic = cvmcriterion(distribution, y_out)

    return test_statistic
end

function validation_cvm_statistic(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata;
    tag_out=nothing,
)

    if isnothing(tag_out)
        tag_out = _validation_tag(data)
    else
        tag_out = _validation_tag(data, tag_out)
    end

    train_data = excludeduration(data, tag_out)

    initialmodel = initialize(pd_type, train_data, 1.0)

    return validation_cvm_statistic(pd_type, data, initialmodel; tag_out = tag_out)
end


# Covariance kernel

"""
    compute_cvm_components(
        fitted_model::MarginalScalingModel,
        train_data::IDFdata,
        d_out::Real,
        ℓ::Integer,
    )

Compute the CDF-gradient function and scaled-information factorization used in
the covariance kernel of the training-validation Cramér--von Mises test.
"""
function compute_cvm_components(
    fitted_model::MarginalScalingModel,
    train_data::IDFdata,
    d_out::Real,
    ℓ::Integer,
)

    d_out > 0 || throw(ArgumentError(
        "The validation duration must be positive, got d_out=$d_out.",
    ))

    ℓ > 0 || throw(ArgumentError(
        "The validation sample size must be positive, got ℓ=$ℓ.",
    ))

    any(d -> isapprox(d, d_out), values(getduration(train_data))) &&
        throw(ArgumentError(
            "The training data must exclude validation duration d_out=$d_out.",
    ))


    T = scalingtype(fitted_model)
    d₀ = duration(fitted_model)
    θ̂ = collect(params(fitted_model))

    # Fitted distribution at the validation duration. It is used only to
    # determine the fixed quantile x = F̂⁻¹(u).
    fitted_distribution = getdistribution(fitted_model, d_out)

    function cdf_gradient(u::Real)

        0 < u < 1 || throw(ArgumentError(
            "The CDF gradient is defined only for 0 < u < 1, got u=$u.",
        ))

        x = quantile(fitted_distribution, u)

        function cdf_at_x(θ::AbstractVector{<:Real})
            model = T(d₀, θ...)
            distribution = getdistribution(model, d_out)

            return cdf(distribution, x)
        end

        return ForwardDiff.gradient(cdf_at_x, θ̂)
    end

    # If H is summed over the training sample, A = H / ℓ is the scaled
    # information matrix entering the covariance kernel.
    H = hessian(fitted_model, train_data)
    A = Symmetric(Matrix(H) / ℓ)
    information_factor = cholesky(A)

    return CvMComponents(
        cdf_gradient,
        information_factor,
    )
end

"""
    approx_eigenvalues(g, A::AbstractMatrix, q::Integer;
        nquad::Integer=max(5q, q + 20),
        eigentol::Real=sqrt(eps(Float64)))

Approximate the `q` largest eigenvalues of the Cramér--von Mises covariance
kernel defined by `g` and `A`, using `nquad` midpoint quadrature nodes.

Eigenvalues smaller than `eigentol` relative to the largest eigenvalue are
treated as numerical zeros.
"""
function approx_eigenvalues(
    cvm_components::CvMComponents,
    q::Integer;
    nquad::Integer=max(5q, q + 20),
    eigentol::Real=sqrt(eps(Float64)),
)

    q > 0 || throw(ArgumentError("q must be positive."))
    nquad >= q || throw(ArgumentError("nquad must be at least q."))
    eigentol >= 0 || throw(ArgumentError("eigentol must be non-negative."))
    isfinite(eigentol) || throw(ArgumentError("eigentol must be finite."))

    nodes = [ (2i - 1) / (2nquad) for i in 1:nquad ]

    # Evaluate g only once at each quadrature node.
    g₁ = cvm_components.cdf_gradient(nodes[1])
    p = length(g₁)

    G = Matrix{Float64}(undef, nquad, p)
    G[1, :] .= g₁

    for i in 2:nquad
        G[i, :] .= cvm_components.cdf_gradient(nodes[i])
    end

    # C[i, j] = g(uᵢ)' A⁻¹ g(uⱼ)
    # Afact = factorize(A)
    C = G * (cvm_components.information_factor \ transpose(G))

    Kmat = Matrix{Float64}(undef, nquad, nquad)

    for j in 1:nquad
        v = nodes[j]

        for i in 1:j
            u = nodes[i]

            Kmat[i, j] = (
                min(u, v) - u * v + C[i, j]
            ) / nquad
        end
    end

    λraw = eigvals(Symmetric(Kmat, :U))

    λmax = maximum(abs, λraw)
    scale = max(λmax, 1.0)

    any(λ -> λ < -eigentol * scale, λraw) &&
        throw(ArgumentError(
            "Negative eigenvalue beyond numerical tolerance.",
        ))

    λ = sort(
        [x for x in λraw if x > eigentol * scale];
        rev=true,
    )

    length(λ) >= q || throw(ArgumentError(
        "Fewer than q positive eigenvalues were found.",
    ))

    return λ[1:q]
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
    scalingtest_bootstrap(
        fitted_model::MarginalScalingModel,
        data::IDFdata;
        tag_out=nothing,
        B::Integer=999,
        rng=Random.default_rng(),
    )

    scalingtest_bootstrap(
        fitted_model::MarginalScalingModel,
        data::IDFdata,
        initialmodel::MarginalScalingModel;
        tag_out=nothing,
        B::Integer=999,
        rng=Random.default_rng(),
    )

Generate `B` dependence-aware bootstrap replicates of the training-validation
Cramér--von Mises test, optionally using a supplied initial model.

Entire yearly rank vectors are resampled to preserve the empirical dependence
across durations. The argument `tag_out` identifies the validation duration,
and `rng` controls the random-number generation.
"""
function scalingtest_bootstrap end

function scalingtest_bootstrap(
    fitted_model::MarginalScalingModel,
    data::IDFdata,
    initialmodel::MarginalScalingModel;
    tag_out = nothing,
    B::Integer = 999,
    rng = Random.default_rng(),
)
    B > 0 || throw(ArgumentError("B must be positive."))

    scalingtype(initialmodel) === scalingtype(fitted_model) ||
          throw(ArgumentError("Fitted model and initial model must be of the same type, got $(scalingtype(fitted_model)) ≠ $(scalingtype(initialmodel))"))

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

    Sstar = Vector{Float64}(undef, B)

    # Generate bootstrap indices sequentially to avoid sharing the RNG across threads.
    bootstrap_indices = [rand(rng, 1:n, n) for _ in 1:B]

    for b in 1:B
        idx = bootstrap_indices[b]
        Ustar = U[idx, :]

        data_star = _idfdata_from_pseudoobs(data_common, fitted_model, Ustar)

        Sstar[b] = validation_cvm_statistic(pd_type, data_star, initialmodel, tag_out = tag_out)
    end

    return Sstar
end

function scalingtest_bootstrap(
    fitted_model::MarginalScalingModel,
    data::IDFdata;
    tag_out=nothing,
    B::Integer=999,
    rng=Random.default_rng(),
)
    B > 0 || throw(ArgumentError("B must be positive."))

    pd_type = scalingtype(fitted_model)

    tags = gettag(data)
    if tag_out === nothing
        tag_out = _validation_tag(data)
    else
        tag_out = _validation_tag(data, tag_out)
    end

    train_data = excludeduration(data, tag_out)
    initialmodel = initialize(pd_type, train_data, 1.0)

    return scalingtest_bootstrap(fitted_model, data, initialmodel; tag_out=tag_out, B=B, rng=rng)
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

