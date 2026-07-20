
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










## Refactor

using CSV
using DataFrames
using Distributions
using IDFCurves
using LinearAlgebra
using Random
using Base.Threads

BLAS.set_num_threads(1)

const d₀ = 1.0
const μ₀ = 20.0
const σ₀ = 5.0
const α = 0.8

const durations = Float64[
    1 / 12,
    1 / 6,
    1 / 4,
    1 / 2,
    1,
    2,
    6,
    12,
    24,
]

const tags = [
    "5min",
    "10min",
    "15min",
    "30min",
    "1h",
    "2h",
    "6h",
    "12h",
    "24h",
]

const duration_dict = Dict(tags .=> durations)




using Pkg
pkg"activate ."

using DataFrames, Distributions, IDFCurves, Random

d₀ = 1.
μ₀ = 20
σ₀ = 5.
ξ = .05
α = .8


tags = tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))


import Base.rand




model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)

@time new_data = rand(model, duration_dict, 100)



function rejection_rate(
    n::Integer,
    ξ::Real,
    M::Integer,
    duration_dict;
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234,
)

    model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)

    # Using the generating model only as the optimizer starting point.
    initialmodel = model

    # One deterministic and independent seed per replication.
    seeds = rand(Xoshiro(seed), UInt64, M)

    # Do not use falses(M), which creates a bit-packed BitVector.
    reject = Vector{Bool}(undef, M)

    Threads.@threads :dynamic for m in eachindex(reject)
        rng = Xoshiro(seeds[m])

        data = rand(rng, model, duration_dict, n)

        T = scalingtest(
            SimpleScaling,
            data,
            initialmodel;
            tag_out="5min",
            q=q,
        )

        reject[m] = IDFCurves.pvalue(T) < level
    end

    return count(identity, reject) / M
end

function run_simulation(
    n_vec,
    ξ_vec,
    M,
    duration_dict;
    q::Integer=20,
    seed::Integer=1234,
)

    N = length(n_vec) * length(ξ_vec)

    results = Vector{
        NamedTuple{
            (:n, :ξ, :RejectionRate),
            Tuple{Int,Float64,Float64},
        }
    }(undef, N)

    k = 0

    for n in n_vec
        for ξ in ξ_vec
            k += 1

            rate = rejection_rate(
                n,
                ξ,
                M,
                duration_dict;
                q=q,
                seed=seed + k,
            )

            results[k] = (
                n=Int(n),
                ξ=Float64(ξ),
                RejectionRate=rate,
            )

            @info "Completed simulation" n ξ rate
        end
    end

    return DataFrame(results)
end

# launch with julia --project=. --threads=auto application.jl

n_vec = [5, 10, 20, 30, 60, 100, 300]
ξ_vec = collect(-0.4:0.1:0.4)
M = 10_000

# Compilation warm-up
rejection_rate(
    first(n_vec),
    first(ξ_vec),
    1,
    duration_dict;
    q=20,
    seed=1,
)

@time results = run_simulation(
    n_vec,
    ξ_vec,
    M,
    duration_dict;
    q=20,
    seed=2026,
)

CSV.write("SimpleScaling_type1_error.csv", results)