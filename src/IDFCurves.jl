module IDFCurves

using CSV, DataFrames, Distributions, ForwardDiff, Gadfly, LinearAlgebra, LogExpFunctions, Optim, PDMats, SpecialFunctions, Extremes, Combinatorics, StatsBase, Logging
import BesselK

import Base: exponent, rand
import Distributions: cdf, dof, location, loglikelihood, logpdf, params, quantile, rand, scale, shape
import Statistics.cor


include("structures.jl")
include("utils.jl")
include("data.jl")
include(joinpath("plots", "plots.jl"))
include(joinpath("plots", "plots_std.jl"))
include("scalingtest.jl")
include("misspecification.jl")


export

    # Variable type
    IDFdata,
    getdata, getduration, gettag, getyear, getKendalldata,

    MarginalScalingModel,

    SimpleScaling,
    GeneralScaling,
    HybridScaling,
    CompositeScaling,
    cdf, duration, exponent, exponent_frequency, getdistribution, getquantile, location, loglikelihood, offset, flattening_offset, params, quantile, quantilecint, rand, scale, shape, params_number,

    DependentScalingModel,
    getcopulatype, getmarginalmodel, getcorrelogram, fit_mle, initialize, cl_bic, bic, parametercint,

    EllipticalCopula,
    GaussianCopula, TCopula, IdentityCopula,

    CorrelationStructure,
    ExponentialCorrelationStructure, MaternCorrelationStructure, UncorrelatedStructure,
    cor,

    # dataitem
    ParamComputation, BaseParamComputation, LinearParamComputation, QuadraticParamComputation, NonDimLinearParamComputation, NonDimQuadraticParamComputation, NonDimExpoParamComputation, FixedSlopeParamComputation,
    Covariate, CovariateStd, Covariates, DataItem, standardize,

    #plots
    qqplot, qqplotci, plotIDFCurves, qqplot_std_data

    #test on scaling models
    scalingtest

end
