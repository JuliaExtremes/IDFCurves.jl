module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim, PDMats, SpecialFunctions
import BesselK

import Base: exponent, rand
import Distributions: cdf, dof, location, loglikelihood, logpdf, params, quantile, rand, scale, shape
import Statistics.cor


include("structures.jl")
include("utils.jl")
include("data.jl")
include("plots.jl")


export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear,

    AbstractScalingModel,

    dGEV,
    SimpleScaling,
    cdf, duration, exponent, getdistribution, location, loglikelihood, offset, params, quantile, quantilecint, rand, scale, shape,

    DependentScalingModel,
    getcopulatype, getmarginalmodel, getcorrelogram,

    EllipticalCopula,
    GaussianCopula, TCopula,

    AbstractCorrelationStructure,
    cor,

    ExponentialCorrelationStructure, MaternCorrelationStructure,

    #plots
    qqplot, qqplotci

end
