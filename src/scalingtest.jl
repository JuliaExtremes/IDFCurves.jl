"""
    scalingtest(fd::Type{<:SimpleScaling}, data::IDFdata;
        d_out::Real = minimum(values(getduration(data))), q::Integer = 100)

Performs the testing procedure in order to state if the model fd may be rejected considering the data. It returns the p-value of the test.
d_out is the duration to be put in the validation set. By default it will be set to the smallest duration in the data.
q is the number of eigenvalues to compute when using the Zolotarev approximation for the p-value.
"""
function scalingtest(pd_type::Type{<:SimpleScaling}, data::IDFdata;
                    d_out::Real = minimum(values(getduration(data))), q::Integer = 100)

    # Parameter estimation
    train_data = excludeduration(data, d_out)
    fitted_model = fit_mle(pd_type, train_data, d_out)

    # Fisher information matrix (normalized)
    hess = hessian(fitted_model, train_data)
    norm_I_Fisher = hess / length(getyear(data, gettag(data,d_out)))

    # Test statistic
    distrib_theo_d_out = getdistribution(fitted_model, d_out) # attention fitted_model doit être un marginalscalingmodel -> à modifier lorsuqe arg sera un DependentScalingModel.
    stat = cvmcriterion(distrib_theo_d_out, getdata(data, gettag(data,d_out)))

    # Kernel function ρ
    g = get_g(fitted_model, d_out)
    ρ(u,v) = minimum([u,v]) - u*v + g(u)' * ( norm_I_Fisher \ g(v) )

    # Eigenvalues
    λs = approx_eigenvalues(ρ, q)

    # Zolotarev approximation

    return
end

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
    get_g(fd::SimpleScaling, d::Float64)

Returns the g function that intervenes in the kernel expression.
The function returns the gradient of the CDF of the model distribution for duration d, evaluated at θ̂ which is the estimated parameter vector
"""
function get_g(fd::SimpleScaling,  d::Real)

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