
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, IDFCurves, GeneralizedChisqDistribution, LinearAlgebra, Test, Random, StatsBase




# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)


df_missing = deepcopy(df)
allowmissing!(df_missing, "5min")
df_missing[1, "5min"] = missing
incomplete_data = IDFdata(df_missing, "Year", duration_dict)



# @time scalingtest(SimpleScaling, data, tag_out = "5min", q=100)
# @time scalingtest(GeneralScaling, data, tag_out = "5min", q=100) 



import IDFCurves: excludeduration, fit_mle, cvmcriterion, hessian, get_g, cvmkernel


function scalingtest(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata,
    q::Integer=100;
    tag_out=nothing,
)

    tag_out = _validation_tag(data, tag_out)

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
    F = getdistribution(fitted_model, d_out)
    S = cvmcriterion(F, y)

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
    λ = approx_eigenvalues(ρ, q)

    # Test statistic distribution
    pd = cvm_distribution(λ)

    return fitted_model, pd, S
end



"""
    approx_eigenvalues(ρ, q::Integer)

Approximate the eigenvalues of the integral operator with kernel `ρ(u, v)` on
`[0, 1]`, using a midpoint Nyström approximation with `q` quadrature points.

The returned eigenvalues are sorted in decreasing order.
"""
function approx_eigenvalues(ρ::K, q::Integer; eigentol::Real=sqrt(eps(Float64))) where {K}
    q > 0 || throw(ArgumentError("q must be positive."))

    Kmat = Matrix{Float64}(undef, q, q)

    for j in 1:q
        v = (2j - 1) / (2q)

        for i in 1:j
            u = (2i - 1) / (2q)
            Kmat[i, j] = ρ(u, v) / q
        end
    end

    λraw = eigvals(Symmetric(Kmat, :U))

    λmax = maximum(abs, λraw)
    scale = max(λmax, 1.0)

    if any(λ -> λ < -eigentol * scale, λraw)
        throw(ArgumentError("Negative eigenvalue beyond numerical tolerance."))
    end

    return reverse(λraw)
end

function cvm_distribution(λ::AbstractVector{<:Real};
    eigentol::Real=sqrt(eps(Float64)))

    isempty(λ) && throw(ArgumentError("At least one positive eigenvalue is required."))

    ν = ones(Int, length(λ))
    δ = zeros(length(λ))

    return GeneralizedChisq(λ, ν, δ, 0.0, 0.0)
end



@time fm, F, S = scalingtest(SimpleScaling, data, 30; tag_out="5min")

fm
F
S

λ = params(F)[1]

quantile(F, 0.95);
@time quantile(F, 0.95)

ccdf(F, S);
@time ccdf(F, S)


import IDFCurves: CvMDistribution

fd = CvMDistribution(λ)
quantile(fd, 0.95);
@time quantile(fd, 0.95)
ccdf(fd, S);
@time ccdf(fd, S)


"""
    scalingtest_bootstrap(pd_type::Type{<:MarginalScalingModel}, data::IDFdata;
        tag_out = nothing, B::Integer = 999, rng = Random.default_rng(),
        return_details::Bool = false)

Compute a dependence-aware bootstrap p-value for the training-validation
Cramér--von Mises goodness-of-fit test.

The bootstrap resamples entire yearly rank vectors across durations. This
preserves the empirical cross-duration dependence while imposing the fitted
scaling model on the marginal distributions.

If `return_details = false`, only the bootstrap p-value is returned. Otherwise,
a named tuple containing the p-value, observed statistic, bootstrap statistics,
fitted model, validation tag, and common years is returned.
"""
function scalingtest_bootstrap(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata;
    tag_out=nothing,
    B::Integer=999,
    rng=Random.default_rng(),
    return_details::Bool=false,
)

    B > 0 || throw(ArgumentError("B must be positive."))

    tags = gettag(data)
    tag_out = _validation_tag(data, tag_out)

    # Restrict the bootstrap to complete years across all durations.
    years = _common_years(data, tags)

    length(years) > 0 || throw(ArgumentError(
        "There is no common year across all durations.",
    ))

    data_common = _restrict_years(data, years; tags=tags)

    # Observed statistic and fitted null model.
    S, fitted_model = _scalingtest_statistic(pd_type, data_common, tag_out)

    # Empirical cross-duration dependence, represented by yearly rank vectors.
    U = _pseudoobs_matrix(data_common, tags)
    n = size(U, 1)

    Sstar = Vector{Float64}(undef, B)

    # Generate bootstrap indices sequentially to avoid sharing the RNG across threads.
    bootstrap_indices = [rand(rng, 1:n, n) for _ in 1:B]

    Threads.@threads for b in 1:B
        idx = bootstrap_indices[b]
        Ustar = U[idx, :]

        data_star = _idfdata_from_pseudoobs(data_common, fitted_model, Ustar, tags)

        Sstar[b], _ = _scalingtest_statistic(pd_type, data_star, tag_out)
    end

    pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)

    if return_details
        return (
            pvalue=pvalue,
            statistic=S,
            bootstrap_statistics=Sstar,
            fitted_model=fitted_model,
            tag_out=tag_out,
            years=years,
        )
    else
        return pvalue
    end
end


"""
    _validation_tag(data::IDFdata, tag_out)

The function returns the tag if it exists in `data`, otherwise, the function throws an error.
"""
function _validation_tag(data::IDFdata, tag_out)

    tags = gettag(data)

    if tag_out in tags
        return tag_out
    else
        throw(ArgumentError("duration tag $tag_out does not correspond to an observed duration."))
    end
end

"""
    _validation_tag(data::IDFdata)

    The tag corresponding to the minimal duration in `data` is returned.   
"""
function _validation_tag(data::IDFdata)

    tags = gettag(data)
    durations = [getduration(data, tag) for tag in tags]
    return tags[argmin(durations)]

end

@testset "_validation_tag" begin

    @test _validation_tag(data, "5min") == "5min"
    @test _validation_tag(data) == "5min"
    @test_throws ArgumentError _validation_tag(data, "nonextistant_tag")

end


"""
    _common_years(data::IDFdata, tags::AbstractVector{<:AbstractString})

Extract the common years vector of the records in `data` corresponding to the `tags`.
"""
function _common_years(data::IDFdata, tags::AbstractVector{<:AbstractString})

    isempty(tags) && throw(ArgumentError("The data set must contain at least one duration."))

    common = Set(getyear(data, tags[1]))

    for tag in tags[2:end]
        intersect!(common, Set(getyear(data, tag)))
    end

    return sort!(collect(common))
end

@testset "_common_years" begin
    @test _common_years(data, gettag(data)) == getyear(data, "5min")
    @test _common_years(incomplete_data, gettag(incomplete_data)) == getyear(incomplete_data, "5min")
end


"""
    _restrict_years(data::IDFdata, years::AbstractVector{<:Integer}; tags = gettag(data))

Return a new `IDFdata` struct restrained to the `years` for each `tags`.    
"""
function _restrict_years(
    data::IDFdata,
    years::AbstractVector{<:Integer};
    tags=gettag(data),
)

    new_tag = String.(collect(tags))

    new_duration = Dict{String,Float64}()
    new_year = Dict{String,Vector{Int64}}()
    new_data = Dict{String,Vector{Float64}}()

    years_vec = Int64.(collect(years))

    for tag in new_tag
        new_duration[tag] = Float64(getduration(data, tag))
        new_year[tag] = copy(years_vec)
        new_data[tag] = [Float64(getdata(data, tag, year)) for year in years_vec]
    end

    return IDFdata(new_tag, new_duration, new_year, new_data)
end

@testset "_restrict_years" begin
    new_data = _restrict_years(data, [2000, 2001])
    @test getyear(new_data, "5min") == [2000, 2001]
end





"""
    _scalingtest_statistic(pd_type::Type{<:MarginalScalingModel}, data::IDFdata, tag_out::String)

Compute the Cramér--von Mises scaling-test statistic for the duration identified by `tag_out`.

### Details

The scaling model is fitted to the training data obtained by removing all observations
from the duration associated with `tag_out`. The fitted model is then used to predict
the marginal distribution at the excluded duration, and the Cramér--von Mises criterion
is computed on the corresponding validation data.

Returns a tuple `(S, fitted_model)`, where `S` is the test statistic and `fitted_model`
is the scaling model fitted without the validation duration.
"""
function _scalingtest_statistic(
    pd_type::Type{<:MarginalScalingModel},
    data::IDFdata,
    tag_out::String,
)

    d_out = getduration(data, tag_out)
    y = getdata(data, tag_out)

    train_data = excludeduration(data, d_out)

    fitted_model = fit_mle(pd_type, train_data, d_out)

    F̂ = getdistribution(fitted_model, d_out)
    S = cvmcriterion(F̂, y)

    return S, fitted_model
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

        # Rank transformation. The denominator n + 1 avoids values exactly
        # equal to 0 or 1, which would create infinite GEV quantiles.
        U[:, j] .= StatsBase.tiedrank(y) ./ (n + 1)
    end

    return U
end


@testset "_pseudoobs_matrix()" begin

    new_tags = ["5min", "10min"]
    new_durations = Dict("5min" => 5/60, "10min" => 10/60)
    new_years = Dict("5min" => collect(2000:2002), "10min" => collect(2000:2002))
    new_obs = Dict("5min" => [3.0, 1.0, 2.0], "10min" => [10.0, 20.0, 20.0])
    new_data = IDFdata(new_tags, new_durations, new_years, new_obs)

    u = _pseudoobs_matrix(new_data, ["5min", "10min"])

    @test u[:, 1] ≈ [3/4, 1/4, 2/4]
    @test u[:, 2] ≈ [1/4, 2.5/4, 2.5/4]

end


function _idfdata_from_pseudoobs(
    template::IDFdata,
    fitted_model::MarginalScalingModel,
    U::AbstractMatrix{<:Real},
    tags::AbstractVector{<:AbstractString},
)

    n, p = size(U)

    p == length(tags) || throw(ArgumentError(
        "The number of columns in U must match the number of duration tags.",
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





res = scalingtest_bootstrap(
    SimpleScaling, data, tag_out="5min", B=8,
    return_details=true
)

@time res = scalingtest_bootstrap(
    SimpleScaling, data, tag_out="5min", B=999,
    return_details=true
)

@time res = scalingtest_bootstrap(
    GeneralScaling, data, tag_out="5min", B=999,
    return_details=true
)









## Replacing the Zolotarev approximation for p-value larger than 1e-8

using GeneralizedChisqDistribution

function _positive_eigenvalues(eigvals::AbstractVector{<:Real};
    eigentol::Real=sqrt(eps(Float64)))

    λraw = collect(float.(eigvals))
    isempty(λraw) && return Float64[]

    λmax = maximum(abs, λraw)
    scale = max(λmax, 1.0)

    if any(λ -> λ < -eigentol * scale, λraw)
        throw(ArgumentError("Negative eigenvalue beyond numerical tolerance."))
    end

    return sort([λ for λ in λraw if λ > eigentol * scale]; rev=true)
end


using Distributions

λ = rand(Gamma(1, 1), 1000)

@time _positive_eigenvalues(λ)





function cvm_distribution(eigvals::AbstractVector{<:Real};
    eigentol::Real=sqrt(eps(Float64)))

    λ = _positive_eigenvalues(eigvals; eigentol=eigentol)

    isempty(λ) && throw(ArgumentError("At least one positive eigenvalue is required."))

    ν = ones(Int, length(λ))
    δ = zeros(length(λ))

    return GeneralizedChisq(λ, ν, δ, 0.0, 0.0)
end

function cvm_pvalue_gchisq(S::Real, eigvals::AbstractVector{<:Real};
    eigentol::Real=sqrt(eps(Float64)))

    S <= 0 && return 1.0

    d = cvm_distribution(eigvals; eigentol=eigentol)

    # return clamp(1 - cdf(d, S), 0.0, 1.0)
    return clamp(ccdf(d, S), 0.0, 1.0)
end


@testset "GeneralizedChisq one-weight test" begin
    λ = 0.7

    for S in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        p_gchisq = cvm_pvalue_gchisq(S, [λ])
        p_exact = ccdf(Chisq(1), S / λ)

        @test p_gchisq ≈ p_exact atol = 1e-6
    end
end

@testset "GeneralizedChisq two unequal positive weights, upper tail" begin
    # Reference values for Q = 0.7χ²₁ + 0.2χ²₁.
    #
    # These values were computed with R using the CompQuadForm package:
    #
    #     library(CompQuadForm)
    #
    #     p <- c(1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
    #
    #     q <- sapply(p, function(pp) {
    #         uniroot(
    #             function(x) {
    #                 CompQuadForm::davies(
    #                     q = x,
    #                     lambda = c(0.7, 0.2),
    #                     h = c(1, 1),
    #                     delta = c(0, 0),
    #                     sigma = 0,
    #                     lim = 50000,
    #                     acc = 1e-12
    #                 )$Qq - pp
    #             },
    #             lower = 0,
    #             upper = 100,
    #             tol = 1e-12
    #         )$root
    #     })
    #
    #     data.frame(q = q, p = p)
    #
    # The reported probabilities are upper-tail probabilities P(Q > q).
end

using Pkg
pkg"activate ."

using DataFrames, Distributions, IDFCurves, GeneralizedChisqDistribution, LinearAlgebra, Test, Random, StatsBase

import IDFCurves.CvMDistribution

    @test_throws ArgumentError CvMDistribution(Float64[])
    @test_throws ArgumentError CvMDistribution([0.0, 0.0])
    @test_throws ArgumentError CvMDistribution([0.7, -0.2])
    @test_throws ArgumentError CvMDistribution([0.7, Inf])
    @test_throws ArgumentError CvMDistribution([0.7, NaN])
    @test_throws ArgumentError CvMDistribution([0.7, 0.2]; eigentol=-1.0)
    @test_throws ArgumentError CvMDistribution([0.7, 0.2]; eigentol=Inf)


    CvMDistribution(Float64[])

   a = Float64.(Float64[])
   isempty(a)

isempty(a) && throw(ArgumentError("At least one eigenvalue is required."))

a = [1., 1.]

any(!isfinite, a) && throw(ArgumentError("All eigenvalues must be finite."))