module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim

import Base: exponent, rand
import Distributions: location, loglikelihood, params, quantile, rand, scale, shape


include("data.jl")
include("structures.jl")
include("plots.jl")

export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear,

    dGEV,
    duration, exponent, getdistribution, location, loglikelihood, offset, params, quantile, quantilevar, rand, scale, shape

end
