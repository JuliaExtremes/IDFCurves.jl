"""
    scalingtest(fd::Type{<:MarginalScalingModel}, data::IDFdata;
        d_out::Real = minimum(values(getduration(data))), q::Integer = 100)

Performs the testing procedure in order to state if the model fd may be rejected considering the data. It returns the p-value of the test.
d_out is the duration to be put in the validation set. By default it will be set to the smallest duration in the data.
q is the number of eigenvalues to compute when using the Zolotarev approximation for the p-value.
"""
function scalingtest(pd_type::Type{<:MarginalScalingModel}, data::IDFdata; tag_out::String, q::Integer = 100)
# function scalingtest(pd_type::Type{<:MarginalScalingModel}, data::IDFdata; tag_out::String=String[], q::Integer = 100)
    @assert tag_out in gettag(data) "duration $tag_out does not correspond to an observed duration."

    d_out = getduration(data, tag_out)

    y = getdata(data, tag_out)
    ℓ = length(y)

    train_data = excludeduration(data, d_out)

    fitted_model = fit_mle(pd_type, train_data, d_out)

    # Test statistic
    F_θ̂ = getdistribution(fitted_model, d_out) #TODO: Make it general for an eventual DependentScalingModel
    S = cvmcriterion(F_θ̂, y)

    # Computing the p-value

    # Fisher information matrix (normalized)
    H = hessian(fitted_model, train_data)
    norm_I_Fisher = H / ℓ

    # Kernel function ρ
    g = get_g(fitted_model, d_out)
    ρ(u,v) = minimum([u,v]) - u*v + g(u)' * ( norm_I_Fisher \ g(v) )

    # Eigenvalues
    λs = approx_eigenvalues(ρ, q) # TODO test if error when Fisher Information Matrix sigular ?

    # Zolotarev approximation
    cdf_approx = zolotarev_approx(λs, S)

    return 1 - cdf_approx
end


# function scalingtest(pd_type::Type{<:MarginalScalingModel}, data::IDFdata; q::Integer = 100)

#     d_out = minimum(getduration.(data, gettag(data)))
#     tag_out = gettag(data, d_out)

#     return scalingtest(pd_type, data, tag_out = tag_out, q=q)

# end

"""
    cvmcriterion(pd::UnivariateDistribution, x::Vector{<:Real})

Returns the Cramer - Von Mises criterion associated to the distribution pd compared to the vector of observations x
"""
function cvmcriterion(pd::UnivariateDistribution, x::Vector{<:Real})
    x̃ = sort(x)
    n = length(x)

    ω² = 1/(12*n) + sum( ((2*i-1)/(2*n) - cdf(pd,x̃[i]) )^2 for i=1:n)

    return ω²

end

"""
    get_g(fd::MarginalScalingModel, d::Float64)

Returns the g function that intervenes in the kernel expression.
The function returns the gradient of the CDF of the model distribution for duration d, evaluated at θ̂ which is the estimated parameter vector
"""
function get_g(fd::MarginalScalingModel,  d::Real)

    pd_type = typeof(fd)
    d₀ = duration(fd)
    θ̂ = collect(params(fd))

    function g(u)
        @assert 0 < u < 1 "g is only defined between 0 and 1"

        x = quantile(getdistribution(fd, d), u) # attention fd doit être un marginalscalingmodel -> à modifier lorsque arg sera un DependentScalingModel.
        function F(θ::AbstractVector{<:Real})
            return cdf( construct_model( pd_type, d₀, map_to_real_space(pd_type,θ) ) , d , x )
        end

        return ForwardDiff.gradient(F, θ̂)
    end

    return g

end

"""
    approx_eigenvalues(ρ::Function, q::Int64)

Approximates the q first (biggest) eigenvalues of the kernel ρ by computing the associated symmetric matrix K.
Returns them in decreasing order.
"""
function approx_eigenvalues(ρ::Function, q::Integer)
    
    K = zeros(Float64,q,q)
    
    for i in 1:q
        for j in 1:q
            K[i,j] = (1/q) * ρ( (2*i-1)/(2*q) , (2*j-1)/(2*q) )
        end
    end

    return reverse( (eigvals(Symmetric(K))) )
end

"""
zolotarev_approx(λs::Vector{<:Real} , x::Real)

Returns an approximation of the CDF of the sum of the λ_i X_i^2, where the X_i^2 are N(0,1), evaluated at x.
The approximation is valid for high quantiles only.
"""
function zolotarev_approx(λs::Vector{<:Real} , x::Real)
    
    @assert length(λs) >= 1 "The vector of λ must contain at elast one element for the Zolotarev approximation to be valid."

    # taking multiplicity into account
    γs = unique(λs)
    multiplicities = [count(==(γ), λs) for γ in γs]

    q = length(γs)
    γ₁ = γs[1]
    
    term1 = - sum([ 0.5 * multiplicities[i] * log(1 - γs[i]/γ₁) for i in 2:q])
    term2 = - log(SpecialFunctions.gamma(0.5 * multiplicities[1]))
    term3 = (0.5 * multiplicities[1] - 1) * log( x/(2*γ₁) )
    term4 = - (x/(2*γ₁)) 

    approx_cdf = maximum([1 - exp(term1 + term2 + term3 + term4), 0 ])

    if term2 < term4 && approx_cdf > .95
        @warn "Zolotarev approximation is outside its validity domain. No conclusion can be made from a small p-value."
    end
        
    return approx_cdf
end