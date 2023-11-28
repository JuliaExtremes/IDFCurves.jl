module IDFCurves

using Distributions

import Base: exponent, rand
import Distributions: location, loglikelihood, rand, scale, shape


include("structures.jl")

export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear,

    dGEV,
    duration, exponent, getdistribution, location, loglikelihood, offset, rand, scale, shape

end
