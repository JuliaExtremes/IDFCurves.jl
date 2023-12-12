using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff

import Distributions.fit_mle

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [20, 5, .04, .76, .07])



initialvalues = [20, 5, .04, .76, .07, 3]



function fit_mle(pd::Type{<:DependentScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractArray{<:Real})

    durations = getduration.(data, gettag(data))
    h = IDFCurves.logdist(durations)

    θ₀ = [IDFCurves.map_to_real_space(dGEV, initialvalues[1:5])..., log(initialvalues[6])]

    model(θ::DenseVector{<:Real}) = DependentScalingModel(
        dGEV(d₀::Real, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
        MvNormal(exp.(-h./exp(θ[6])))
    )

    fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

    @time res = Optim.optimize(fobj, θ₀)

    θ̂ = Optim.minimizer(res)

end

fit_mle(DependentScalingModel, data, 1, initialvalues)



struct covariogram

 end


covmatrix()

function corfun(-d/ρ)



durations = getduration.(data, gettag(data))
h = IDFCurves.logdist(durations)


θ₀ = [IDFCurves.map_to_real_space(dGEV, params(fd))..., .8]

model(θ::DenseVector{<:Real}) = DependentScalingModel(
    dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
    MvTDist(15, exp.(-h./exp(θ[6])))
    )

fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

@time res = Optim.optimize(fobj, θ₀)

θ̂ = Optim.minimizer(res)



fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data, autodiff=:forward)
H = ForwardDiff.hessian(fobj, θ̂)




θ₀ = [IDFCurves.map_to_real_space(dGEV, params(fd))..., .046, .8]

model(θ::DenseVector{<:Real}) = DependentScalingModel(
    dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
    MvTDist(15, IDFCurves.matern.(h, exp(θ[6]), exp(θ[7])))
    )


fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

# res = Optim.optimize(TwiceDifferentiable(fobj, θ₀; autodiff = :forward), θ₀)
@time res = Optim.optimize(fobj, θ₀)

θ̂ = Optim.minimizer(res)


# Ça fonctionne
model(θ::DenseVector{<:Real}) = dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...)
fobj2(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)
ForwardDiff.gradient(fobj2, θ̂[1:5])


tags = gettag(data)
idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
d = getduration.(data, tags)

y = getdata.(data, tags, idx')

M = dGEV(1, IDFCurves.map_to_param_space(dGEV, θ̂[1:5])...)

u = cdf.(M, d, y)



C = MvTDist(15, IDFCurves.matern.(h, exp(θ̂[6]), exp(θ̂[7])))

IDFCurves.logpdf_TCopula(C, u[:, 1])

f(θ::AbstractVector{T} where T) = -IDFCurves.logpdf_TCopula( MvTDist(15, IDFCurves.matern.(h, exp(θ[1]), exp(θ[2]))), u[:, 1], autodiff=:forward)

ForwardDiff.gradient(f, θ̂[6:7])


function f2(θ::AbstractVector{T}) where T

    return -IDFCurves.logpdf_TCopula( MvTDist(15, IDFCurves.matern.(h, exp(θ[1]), exp(θ[2]))), u[:, 1], autodiff=:forward)

end


f2(θ̂[6:7])
ForwardDiff.gradient(f2, θ̂[6:7])


function f3(θ)

    # C = MvTDist(15, IDFCurves.matern.(h, exp(θ[1]), exp(θ[2])))

    Σ = exp.(-h/exp(θ))
    # C = MvNormal(Σ)
    # x = quantile.(Normal(), u[:,1])
    

    C = MvTDist(15, Σ)
    x = quantile.(TDist(15), u)

    # C = MvNormal(IDFCurves.matern.(h, exp(θ[1]), exp(θ[2])))
    


    ll = -logpdf(C, x)

end

f3(θ̂[6])
ForwardDiff.derivative(f3, θ̂[6])




function f4(θ)
    
    Σ = IDFCurves.matern.(h, exp(θ[1]), exp(θ[2]))

    C = MvNormal(Σ)
    x = quantile.(Normal(), u[:,1])

    ll = -logpdf(C, x)

end
f4(θ̂[6:7])
ForwardDiff.gradient(f4, θ̂[6:7])










model(θ::DenseVector{<:Real}) = DependentScalingModel(
    dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
    MvTDist(15, IDFCurves.matern.(h, exp(θ[6]), exp(θ[7])))
    )
fobj3(θ::DenseVector{<:Real}) = -loglikelihood(IDFCurves.getmarginalmodel(model([θ..., .043, .8])), data)
ForwardDiff.gradient(fobj3, θ̂[1:5])



fobj3(θ::DenseVector{<:Real}) = -loglikelihood2(model(θ), data)
ForwardDiff.gradient(fobj3, θ̂)

function loglikelihood2(pd::DependentScalingModel, data)

    # tags = gettag(data)
    # idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
    # d = getduration.(data, tags)

    # y = getdata.(data, tags, idx')

    # Marginal loglikelihood
    ll = loglikelihood(IDFCurves.getmarginalmodel(pd), data)

    # # Copula loglikelihood #TODO Check for other type of elliptical copula
    # u = cdf.(getmarginalmodel(pd), d, y)
    # for c in eachcol(u)
    #     ll += IDFCurves.logpdf_TCopula(getcopula(pd), c)
    # end

    return ll

end









H = ForwardDiff.hessian(fobj, θ̂)









# θ₀ = [IDFCurves.map_to_real_space(dGEV, params(fd))..., .046, .8]

θ₀ = [IDFCurves.map_to_real_space(dGEV, params(fd))...]

model(θ::DenseVector{<:Real}) = DependentScalingModel(
    dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
    MvTDist(15, IDFCurves.matern.(h, exp(θ[6]), exp(θ[7])))
    )


fobj(θ::DenseVector{<:Real}) = -f(model([θ..., .046, .8]), data)




function f(pd::DependentScalingModel, data::IDFdata)

    marginalmodel = getmarginalmodel(pd)
    
    θ̂ = params(marginalmodel) 

    ll = loglikelihood(marginalmodel, data)

end



ForwardDiff.gradient(fobj, θ₀)


















function fit_mle_gradient_free(pd::Type{DependentScalingModel}, data::IDFdata, d₀::Real, initialvalues::AbstractVector{<:Real})

    θ₀ = [IDFCurves.map_to_real_space(typeof(IDFCurves.getmarginalmodel(pd)), params(fd))..., 
        log(initialvalues[6]), log(initialvalues[7])]

    # fobj(θ::DenseVector{<:Real}) = -loglikelihood(dGEV(d₀, map_to_param_space(dGEV, θ)...), data)
    
    # res = Optim.optimize(fobj, θ₀)
    
    # if Optim.converged(res)
    #     θ̂ = Optim.minimizer(res)
    # else
    #     @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
    #     θ̂ = θ₀
    # end
    
    # fd = dGEV(d₀, map_to_param_space(dGEV, θ̂)...)
    
    # return fd
end


θ₀ = [IDFCurves.map_to_real_space(dGEV, params(fd))..., .046, .8]

model(θ::DenseVector{<:Real}) = DependentScalingModel(
    dGEV(1, IDFCurves.map_to_param_space(dGEV, θ[1:5])...),
    MvTDist(15, IDFCurves.matern.(h, exp(θ[6]), exp(θ[7])))
    )


fobj(θ::DenseVector{<:Real}) = -loglikelihood(model(θ), data)

# res = Optim.optimize(TwiceDifferentiable(fobj, θ₀; autodiff = :forward), θ₀)
@time res = Optim.optimize(fobj, θ₀)

θ̂ = Optim.minimizer(res)


function loglikelihood_ad(pd::DependentScalingModel, data)

    # tags = gettag(data)
    # idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
    # d = getduration.(data, tags)

    # y = getdata.(data, tags, idx')

    # Marginal loglikelihood
    ll = loglikelihood(getmarginalmodel(pd), data)

    # Copula loglikelihood #TODO Check for other type of elliptical copula
    # u = cdf.(getmarginalmodel(pd), d, y)
    # for c in eachcol(u)
    #     ll += IDFCurves.logpdf_TCopula_ad(getcopula(pd), c)
    # end

    return ll

end

f(θ::DenseVector{<:Real}) = -loglikelihood_ad(model(θ), data)

H = ForwardDiff.hessian(f, θ̂)

f(p) = IDFCurves.quantile_ad(TDist(5), p)
ForwardDiff.derivative(f, .25)

f2(p) = quantile(TDist(5), p)
ForwardDiff.derivative(f2, .25)


fd = IDFCurves.fit_mle_gradient_free(dGEV, data, 1, [1, 1, 0, .9, 1])


h = IDFCurves.logdist(durations)
Σ = IDFCurves.matern.(h, 5, 1) 
C = MvTDist(15, Σ)

fmm = DependentScalingModel(fd, C)









using LinearAlgebra, Distributions, Test, PDMats
import Distributions: logpdf, dof


abstract type EllipticalCopula end

struct GaussianCopula <: EllipticalCopula
    cormatrix::PDMat
     
    GaussianCopula(Σ::AbstractMatrix{<:Real}) = new(PDMat(Σ))

end

function getcormatrix(obj::EllipticalCopula)
    return obj.cormatrix
end

@testset "GaussianCopula constructor" begin
    C = GaussianCopula([1 0; 0 1])
    @test getcormatrix(C) == PDMat([1 0; 0 1])
end

function logpdf(C::GaussianCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    D = MvNormal(getcormatrix(C))
    
    x = quantile.(Normal(), u)

    return logpdf(D, x) - sum(logpdf.(Normal(), x))

end

@testset "logpdf(::GaussianCopula)" begin
    C = GaussianCopula([1 .5; .5 1])
    @test logpdf(C, [.25, .2]) ≈ 0.32840717930070484 # Value from Copulas.jl
end



struct TCopula <: EllipticalCopula
    df::Real
    cormatrix::PDMat
     
    TCopula(df::Real, Σ::AbstractMatrix{<:Real}) = new(df, PDMat(Σ))

end

function dof(C::EllipticalCopula)
    return C.df
end

function logpdf(C::TCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    ν = dof(C)
    D = MvTDist(ν, getcormatrix(C))
    
    x = quantile.(TDist(ν), u)

    return logpdf(D, x) - sum(logpdf.(TDist(ν), x))

end

@testset "TCopula constructor" begin
    C = TCopula(5, [1 0; 0 1])
    @test dof(C) == 5
    @test getcormatrix(C) == PDMat([1 0; 0 1])
end


@testset "logpdf(::TCopula)" begin
    C = TCopula(5, [1 .5; .5 1])
    @test logpdf(C, [.25, .2]) ≈0.409866913801614 # Value from Copulas.jl
end





# struct EllipticalCopula{T <: ContinuousUnivariateDistribution}
#     cormatrix::Hermitian
# end

# struct EC{T::Type{<:ContinuousUnivariateDistribution}}
#     cormatrix::Hermitian
# end


# EllipticalCopula{Normal}(Hermitian([1 0; 0 1]))




function getcormatrix()

@testset
    GaussianCopula([1 0; 0 1])  