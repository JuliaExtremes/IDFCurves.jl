module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim, PDMats, SpecialFunctions

import Base: exponent, rand
import Distributions: cdf, dof, location, loglikelihood, logpdf, params, quantile, rand, scale, shape


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
    cdf, duration, exponent, getdistribution, location, loglikelihood, offset, params, quantile, quantilecint, rand, scale, shape,

    DependentScalingModel,
    getcopula, getmarginalmodel,

    EllipticalCopula,
    GaussianCopula, TCopula,

    #plots
    qqplot, qqplotci

end
