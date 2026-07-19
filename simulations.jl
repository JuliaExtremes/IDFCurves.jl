
using Pkg
pkg"activate ."

using DataFrames, Distributions, IDFCurves

d₀ = 1.
μ₀ = 20
σ₀ = 5.
ξ = .05
α = .8


tags = tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))


import Base.rand

function rand(model::MarginalScalingModel, duration_dict::Dict{String, <:Real}, n::Int = 1)

    tags = collect(keys(duration_dict))
    year_index = collect(1:n)

    year_dict = Dict(zip(tags, repeat([year_index], length(tags))))
    data_dict = Dict{String, Vector{Float64}}()

    for tag in tags
        d = duration_dict[tag]
        (d>0) || throw(ArgumentError("Duration corresponding to tag = $tag must be positive, got $d."))
        pd = IDFCurves.getdistribution(model, d)
        data_dict[tag] = rand(pd, n)
    end

    return IDFdata(tags, duration_dict, year_dict, data_dict)

end


# @time new_data = rand(model, duration_dict, 100)

df = DataFrame(n=Int64[], ξ=Float64[], RejectionRate=Float64[])

n_vec = [5, 10, 20, 30, 60, 100, 300]
ξ_vec = collect(-.4:.1:.4)

M = 5

for n in n_vec
    for ξ in ξ_vec
        reject = falses(M)
        for m in 1:M
            model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
            data = rand(model, duration_dict, n)
            T = scalingtest(SimpleScaling, data, tag_out="5min")
            reject[m] = IDFCurves.pvalue(T) < 0.05
        end
        push!(df, [n, ξ, count(reject)/M])
    end
end

CSV.write("SimpleScaling_type1_error.csv", df)