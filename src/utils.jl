
"""
    compute_coeff(pd::TDist, max_coeff::Int=750)
    
Return the coefficients for the series development of the Student distribution in the bulk.
"""
function compute_coeff(pd::TDist, max_coeff::Int=750)

    ν = dof(pd) 

    c = Vector{Float64}(undef, max_coeff)

    c[1] = 1.

    i = 1

    while abs(c[i]) > 2*eps() && i<= max_coeff
    
        s = 0.
        for k in 0:(i-1)
            for l in  0:(i -k - 1)
                factor = (1 + 1/ν) * (2*l +1)*(2*i - 2*k - 2*l - 1)  - 2*k*(2*k +1)/ ν
                s += c[k + 1]*c[l+ 1]*c[i-k-l-1 + 1] * factor
            end
        end

        c[i+1] = s/(2*i*(2*i +1))

        i += 1
    
    end

    return c[1:i]

end

"""
    compute_tail_coeff(pd::TDist, max_coeff::Int=750)
    
Return the coefficients for the series development of the Student distribution in the tails.
"""
function compute_tail_coeff(pd::TDist, max_coeff::Int=750)

    ν = dof(pd)

    c = Vector{Float64}(undef, max_coeff)
 
    c[1] = 1.

    i = 2

    while abs(c[i-1]) > 2*eps() && i <= max_coeff
        
        s = 0.
        
        for k in 1:(i-1)
            for m in 1:(i-k)
                factor = k*(k - ν/2) + (ν/2 - 3/2)*m*(i+1 - k - m)
                s += c[k]*c[m]*c[i+1 - k- m] * factor
            end
        end

        if i > 2
            for k in 2:i-1
                s += c[k]*c[i+1-k] * ((1-ν/2)*k*(i-k) - k*(k-1))
            end
        end

        c[i] = s / (i^2 + i * (ν/2 - 2) + (1 - ν/2))

        i+=1
    end

    return c[1:i-1]
end

"""
    logpdf_TCopula(C::Distributions.GenericMvTDist, u::AbstractVector{<:Real})

Logpdf of the Student copula `C` evaluated at `u`.
"""
function logpdf_TCopula(C::Distributions.GenericMvTDist, u::AbstractVector{T} where T; autodiff::Symbol=:none)
    @assert all(0 .≤ u .≤ 1) 
    @assert autodiff in (:none, :forward)
    
    ν = C.df
    margdist = TDist(ν)

    if autodiff == :none
        x = quantile.(margdist, u)
    else
        x = quantile_ad.(margdist, u)
    end

    return logpdf(C, x) - sum(logpdf.(margdist, x))

end

"""
    matern(d::Real, ν::Real, ρ::Real)

Return the Matérn correlation of parameter (ν, ρ) between measurements taken at two points separated by the distance `d`.

### Details
[Matérn covariance function](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
"""
function matern(d::Real, ν::T where T, ρ::T where T)
    @assert d ≥ 0 "distance must be non-negative."
    @assert ν > 0 "ν must be positive."
    @assert ρ > 0 "ρ must be positive."

    if d ≈ 0.
        c = 1.
    else
        c = 2^(1-ν)/SpecialFunctions.gamma(ν) * (sqrt(2*ν) * d/ρ)^ν * SpecialFunctions.besselk(ν,sqrt(2*ν)*d/ρ)
    end

    return c

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

"""
    quantile_ad(pd::TDist, p::Real)

Compute the quantile of level `p` of the Student distribution `pd` with an implementation compatible with automatic differentiation.
"""
function quantile_ad(pd::TDist, p::Real)
    @assert  0<p<1 "the quantile level must be in (0,1)."

    ν = dof(pd)

    if ν ≈ 2
        q = quantile_TDist2(pd, p)
    elseif ν ≈ 4
        q = quantile_TDist4(pd, p)
    else
        if p < .2
            q = quantile_TDist_ltail(pd, p)
        elseif p > .8
            q = quantile_TDist_rtail(pd, p)
        else
            q = quantile_TDist(pd, p)
        end
    end

    return q

end


function quantile_TDist2(pd::TDist, p::Real)
    @assert 0<p<1 "the quantile level must be in (0,1)."

    ν = dof(pd)

    @assert ν ≈ 2 "this approximation is only valid for ν = 2."

    α = 4*p*(1-p)
    q = 2 * (p - 1 /2) * sqrt(2 / α)

    return q

end

function quantile_TDist4(pd::TDist, p::Real)
    @assert 0<p<1 "the quantile level must be in (0,1)."

    ν = dof(pd)

    @assert ν ≈ 4 "this approximation is only valid for ν = 4."

    α = 4*p*(1-p)
    β = cos(acos(sqrt(α))/3) / sqrt(α)
    if p >= 1/2
        q = 2 * sqrt(β - 1)
    else
        q = -2 * sqrt(β - 1)
    end

    return q

end

function quantile_TDist(pd::TDist, p::Real)
    @assert .2 ≤ p ≤ .8 "the approximation is precise for p in [0.2 , 0.8]."

    ν = dof(pd)

    v = (p - 1/2) * √(ν * π) * gamma(ν/2)/gamma((ν+1)/2)

    c = compute_coeff(pd)

    x = 0:length(c)-1

    q = sum( v .^(2 .* x .+ 1 ) .* c)

    return q
end

function quantile_TDist_rtail(pd::TDist, p::Real)
    @assert p>.8 "the approximation is precise for p > 0.8."

    ν = dof(pd)

    c = compute_tail_coeff(pd)

    x_β = (ν * (1-p) * SpecialFunctions.beta(ν/2, 0.5) ) ^ (2/ν)

    x = 1:length(c)

    w_β = sum(c .* x_β .^ x)

    q = √(ν* (1/w_β - 1)) 

    return q
end

function quantile_TDist_ltail(pd::TDist, p::Real)
    @assert p<.2 "the approximation is precise for p < 0.2."

    q = quantile_TDist_rtail(pd, 1-p)

    return -q
end



"""
    compute_derivatives(g::function)

Computes and returns the functions associated to the gradient and the hessian of function g. Those functions are compatible with Optim.jl.
"""
function compute_derivatives(g::Function)

    function grad_g(G, θ)
        grad = ForwardDiff.gradient(g, θ)
        for i in eachindex(G)
            G[i] = grad[i]
        end
    end
    function hessian_g(H, θ)
        hess = ForwardDiff.hessian(g, θ)
        for i in axes(H,1)
            for j in axes(H,2)
                H[i,j] = hess[i,j]
            end
        end
    end

    return grad_g, hessian_g

end