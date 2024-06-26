module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim, PDMats, SpecialFunctions, Extremes, Combinatorics, StatsBase
import BesselK

import Base: exponent, rand
import Distributions: cdf, dof, location, loglikelihood, logpdf, params, quantile, rand, scale, shape
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
    cdf, duration, exponent, getdistribution, location, loglikelihood, offset, params, quantile, quantilecint, rand, scale, shape, params_number,

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
