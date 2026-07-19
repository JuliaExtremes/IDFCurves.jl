
# julia --threads auto script.jl

using Pkg
pkg"activate ."

using DataFrames, Distributions, Extremes, IDFCurves, LinearAlgebra, Optim
using Cairo, Gadfly, Fontconfig

using Test

## Application

# data at Mtl Trudeau
df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Mtl_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Mtl_cdf_generalscaling.pdf"), fig)


# data at Nanaimo
df = IDFCurves.dataset("1025369")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

plotIDFCurves(fd, data)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out="5min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data, B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)


train_data = IDFCurves.excludeduration(data, "5min")
q, _ = Extremes.ecdf(getdata(data, "5min"))

ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 250,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Nan_cdf_simplescaling.pdf"), fig)




# data at Vancouver (Cette station nécessite le universalscaling !)
df = IDFCurves.dataset("1108446")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

# Fit the Simple Scaling model 
fd = IDFCurves.fit_mle(SimpleScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(SimpleScaling, data, tag_out = "10min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data; tag_out = "10min", B = B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)

# Fit the General Scaling model
fd = IDFCurves.fit_mle(GeneralScaling, data, 1.)

# Goodness-of-fit test using the 5min duration as validation
T = scalingtest(GeneralScaling, data, tag_out="10min")
S = T.test_statistic
pvalue = IDFCurves.pvalue(T)
reject = IDFCurves.decision(T)

# Adjust the p-value for the interduration dependence
B = 999 # Number of bootstrap samples
Tstar = IDFCurves.scalingtest_bootstrap(fd, data; tag_out = "10min", B=B)
Sstar = [Tstar[b].test_statistic for b in eachindex(Tstar)]
adjusted_pvalue = (1 + count(s -> s >= S, Sstar)) / (B + 1)



train_data = IDFCurves.excludeduration(data, "10min")
q, _ = Extremes.ecdf(getdata(data, "10min"))

function F(q::AbstractVector{<:Real}, x::Real)

    # issorted(q) || throw(ArgumentError("quantiles must be sorted"))
    return count(q .≤ x) / (length(q) + 1.)

end


ss = IDFCurves.fit_mle(SimpleScaling, train_data, 1)

pd = getdistribution(ss, 10/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("10-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Van_cdf_simplescaling.pdf"), fig)


gs = IDFCurves.fit_mle(GeneralScaling, train_data, 1)

pd = getdistribution(gs, 5/60)

fig = plot([y->cdf(pd, y), y->F(q, y)], 0, 150,
    Guide.xlabel("5-min precipitation intensity (mm/h)"),
    Guide.ylabel("probability"),
    Theme(key_position=:none)
)

# draw(PDF("Mtl_cdf_generalscaling.pdf"), fig)



## approx_eigenvalues performances

using Pkg
pkg"activate ."

using DataFrames, Distributions, Extremes, IDFCurves, LinearAlgebra, Optim
using Cairo, Gadfly, Fontconfig

using Test

import IDFCurves.approx_eigenvalues

# Old version
@testset "approx_eigenvalues()" begin

    # Brownian bridge kernel:
    # ρ(u, v) = min(u, v) - u*v
    #
    # It is obtained from cvmkernel by setting the correction term to zero.
    g(u) = [0.0]
    A = Matrix{Float64}(I, 1, 1)

    ρ = IDFCurves.cvmkernel(g, A)

    λs = IDFCurves.approx_eigenvalues(ρ, 10)

    @test length(λs) == 10
    @test issorted(λs; rev = true)
    @test all(λs .>= -sqrt(eps(Float64)))

    λs = IDFCurves.approx_eigenvalues(ρ, 100)

    @test length(λs) == 100
    @test issorted(λs; rev = true)

    # Exact eigenvalues of the Brownian bridge covariance kernel
    @test isapprox(λs[1], 1 / π^2, rtol = 2e-3)
    @test isapprox(λs[2], 1 / (4π^2), rtol = 3e-3)
    @test isapprox(λs[3], 1 / (9π^2), rtol = 5e-3)
end

# New version
@testset "approx_eigenvalues()" begin

    # Brownian bridge kernel:
    # ρ(u, v) = min(u, v) - u*v
    #
    # It is obtained from cvmkernel by setting the correction term to zero.
    g(u) = [0.0]
    A = Matrix{Float64}(I, 1, 1)

    λs = IDFCurves.approx_eigenvalues(g, A, 10)

    @test length(λs) == 10
    @test issorted(λs; rev = true)
    @test all(λs .>= -sqrt(eps(Float64)))

    # Exact eigenvalues of the Brownian bridge covariance kernel
    @test isapprox(λs[1], 1 / π^2, rtol = 2e-3)
    @test isapprox(λs[2], 1 / (4π^2), rtol = 3e-3)
    @test isapprox(λs[3], 1 / (9π^2), rtol = 5e-3)

end

using ForwardDiff
import IDFCurves: scalingtype, hessian, approx_eigenvalues

struct CvMComponents{G,F}
    cdf_gradient::G
    information_factor::F
end

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

    !any(values(getduration(train_data)) .≈  d_out) || throw(ArgumentError("Fitted model and data must exclude duration d_out = $d_out."))


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








df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

tag_out = "5min"
train_data = IDFCurves.excludeduration(data, tag_out)
fm = IDFCurves.fit_mle(SimpleScaling, train_data, 1.)
ℓ = length(getdata(data, tag_out))




@time cvm_components = compute_cvm_components(fm, train_data, 5/60, ℓ)
@time λ = approx_eigenvalues(cvm_components, 20)



# Verifying if approx_eigenvalues are the same (YES IT WORKS)

d_out = 5/60

H = hessian(fm, train_data)
A = Symmetric(Matrix(H) / ℓ)
information_factor = cholesky(A)

g = IDFCurves.get_g(fm, d_out)
ρ = IDFCurves.cvmkernel(g, A)

@time λ̃ =  approx_eigenvalues(ρ, 20)

λ .≈ λ̃





## Implementation


using Pkg
pkg"activate ."

using DataFrames, IDFCurves

using Test

df = IDFCurves.dataset("702S006")
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
data = IDFdata(df, "Year", duration_dict)

tag_out = "5min"

T = scalingtest(SimpleScaling, data, tag_out=tag_out)
IDFCurves.pvalue(T)

T = scalingtest(GeneralScaling, data, tag_out=tag_out)
IDFCurves.pvalue(T)