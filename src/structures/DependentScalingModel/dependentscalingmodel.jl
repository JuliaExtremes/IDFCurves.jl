
struct DependentScalingModel
    marginal::ContinuousMultivariateDistribution
    copula::ContinuousMultivariateDistribution
end

function getcopula(pd::DependentScalingModel)
    return pd.copula
end

function getmarginalmodel(pd::DependentScalingModel)
    return pd.marginal
end

function loglikelihood(pd::DependentScalingModel, data)

    tags = gettag(data)
    idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
    d = getduration.(data, tags)

    y = getdata.(data, tags, idx')

    # Marginal loglikelihood
    ll = loglikelihood(getmarginalmodel(pd), data)

    # Copula loglikelihood #TODO Check for other type of elliptical copula
    u = cdf.(getmarginalmodel(pd), d, y)
    for c in eachcol(u)
        ll += IDFCurves.logpdf_TCopula(getcopula(pd), c)
    end

    return ll

end