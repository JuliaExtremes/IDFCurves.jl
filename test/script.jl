using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions

import IDFCurves.loglikelihood

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)

fd = IDFCurves.fit_mle(dGEV, data, 1, [1, 1, 0, .9, 1])


h = IDFCurves.logdist(durations)
Σ = IDFCurves.matern.(h, 5, 1) 
C = MvTDist(15, Σ)

fmm = DependentScalingModel(fd, C)






function loglikelihood(pd::DependentScalingModel, data)

    tags = gettag(data)
    idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
    d = getduration.(data, tags)

    y = getdata.(data, tags, idx')

    # Marginal loglikelihood
    ll = loglikelihood(getmarginalmodel(pd), data)

    # Copula loglikelihood #TODO Check for other type of elliptical copula
    u = cdf.(getmarginalmodel(pd), durations, y)
    for c in eachcol(u)
        ll += IDFCurves.logpdf_TCopula(getcopula(pd), c)
    end

    return ll

end

@testset "loglikelihood(::DependentScalingModel)" begin
    
    tags = ["1h", "2h"]
    durations = [1., 2.]
    years = [2020, 2021]
    y = hcat(2:3, 0:1)

    d1 = Dict(zip(tags, durations))
    d2 = Dict(tags[1] => years, tags[2] => years)
    d3 = Dict(tags[1] => y[:,1], tags[2] => y[:,2])

    data = IDFdata(tags, d1, d2, d3)

    mm = dGEV(1, 1, 1, 0, .8, .5)
    C = MvTDist(15, [1. .5; .5 1])
    pd = DependentScalingModel(mm, C)

    @test loglikelihood(pd, data) ≈ -6.330260155320674

end





@time loglikelihood(pd, data)



tags = gettag(data)
idx = getyear(data, tags[1]) #TODO Check for missing data (assumes that all years are present for each duration)
d = getduration.(data, tags)

y = getdata.(data, tags, idx')

ll = loglikelihood(getmarginalmodel(pd), data)

# Copula loglikelihood
u = cdf.(getmarginalmodel(pd), durations, y)

for c in eachcol(u)
    ll += IDFCurves.logpdf_TCopula(getcopula(pd), c)
end




tags = gettag(data)
idx = getyear(data, "5min") #TODO Check for missing data

d = getduration.(data, tags)


y = getdata.(data, tags, idx')

u = cdf.(fd, durations, y)

ll = loglikelihood(fd, data)

for c in eachcol(u)
    ll += IDFCurves.logpdf_TCopula(C, c)
end

logpdf_TCopula(C, u)





y₁ = y[1]

u₁ = cdf()




function logpdf(pd::DependentScalingModel, data::IDFdata)

    

end








m = length(C)
u = rand(m)


@time logpdf_TCopula(C, u)

uv = [rand(m) for i in 1:10]

@time logpdf_TCopula.(Ref(C), uv)







# Construct the correlation matrix
IDFCurves.matern.([0. 2;2 0.],1,1)



