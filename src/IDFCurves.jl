module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim, SpecialFunctions

import Base: exponent, rand
import Distributions: cdf, location, loglikelihood, params, quantile, rand, scale, shape


include("structures.jl")
include("utils.jl")
include("data.jl")
include("plots.jl")


export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear,

    dGEV,
    cdf, duration, exponent, getdistribution, location, loglikelihood, offset, params, quantile, quantilecint, rand, scale, shape,

    #plots
    qqplot, qqplotci

end
