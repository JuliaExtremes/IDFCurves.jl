module IDFCurves

using CSV, DataFrames, Distributions, Gadfly, LogExpFunctions, Optim

import Base: exponent, rand
import Distributions: location, loglikelihood, params, rand, scale, shape


include("structures.jl")
include("plots.jl")

export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear,

    dGEV,
    duration, exponent, getdistribution, location, loglikelihood, offset, params, rand, scale, shape

end
