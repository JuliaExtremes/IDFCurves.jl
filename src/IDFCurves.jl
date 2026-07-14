module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim, PDMats, SpecialFunctions, Extremes, Combinatorics, StatsBase
import BesselK, QuadGK, Random

import Base: exponent, rand
import Distributions: ContinuousUnivariateDistribution, cdf, ccdf,  dof, mean, location, loglikelihood, logpdf, params, quantile, rand, scale, shape, std, var
import Statistics.cor

include("structures.jl")
include("utils.jl")
include("data.jl")
include("plots.jl")
include("scalingtest.jl")
include("misspecification.jl")


export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear, getKendalldata,

    MarginalScalingModel,

    SimpleScaling,
    GeneralScaling,
    UniversalScaling,
    cdf, duration, exponent, getdistribution, location, loglikelihood, offset, largescale_offset, params, quantile, quantilecint, rand, scale, shape, params_number,

    DependentScalingModel,
    getcopulatype, getmarginalmodel, getcorrelogram, fit_mle, initialize,

    EllipticalCopula,
    GaussianCopula, TCopula, IdentityCopula,

    CorrelationStructure,
    ExponentialCorrelationStructure, MaternCorrelationStructure, UncorrelatedStructure,
    cor,

    #plots
    qqplot, qqplotci, plotIDFCurves,

    #test on scaling models
    scalingtest

end
