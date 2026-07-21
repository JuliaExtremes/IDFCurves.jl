
# julia --project=. --threads=auto simulations.jl

# using Pkg
# pkg"activate ."

using CSV, DataFrames, Distributions, IDFCurves, LinearAlgebra, Random
using Base.Threads

BLAS.set_num_threads(1)

# Simulation parameters
n_vec = [10, 15, 30, 50, 75, 100]
ξ_vec = collect(-0.4:0.2:0.4)
simulation_size = 1000


d₀ = 1.
μ₀ = 20
σ₀ = 5.
ξ = .05
α = .8
δ = .05

tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))

model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
template = rand(model, duration_dict, 1)

function rejection_rate(target_model::Type{<:MarginalScalingModel}, 
    model::MarginalScalingModel,
    template::IDFdata,
    sample_size::Integer,
    simulation_size::Integer;
    tag_out = nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234)

    if tag_out === nothing
        tag_out = IDFCurves._validation_tag(template)
    else
        tag_out = IDFCurves._validation_tag(template, tag_out)
    end

    duration_dict = getduration(template)

    # Using the generating model only as the optimizer starting point.
    if target_model === SimpleScaling
        initialmodel = IDFCurves.scalingtype(model)(duration(model), location(model), scale(model), 0., exponent(model))
    else
        initialmodel = IDFCurves.scalingtype(model)(duration(model), location(model), scale(model), 0., exponent(model), offset(model))
    end

    # One deterministic and independent seed per replication.
    seeds = rand(Xoshiro(seed), UInt64, simulation_size)

    # Do not use falses(M), which creates a bit-packed BitVector.
    reject = Vector{Bool}(undef, simulation_size)

    Threads.@threads :dynamic for m in eachindex(reject)
        rng = Xoshiro(seeds[m])

        data = rand(rng, model, duration_dict, sample_size)

        T = scalingtest(target_model, data, initialmodel; tag_out=tag_out, q=q)

        reject[m] = IDFCurves.pvalue(T) < level
    end

    return count(identity, reject) / simulation_size
end

function run_simulation_simplescaling_errortype1(
    n_vec::AbstractVector{<:Integer},
    ξ_vec::AbstractVector{<:Real},
    simulation_size::Integer,
    template::IDFdata;
    tag_out = nothing,
    q::Integer=20,
    level::Real = .05,
    seed::Integer=1234)

    N = length(n_vec) * length(ξ_vec)

    results = Vector{NamedTuple{(:n, :ξ, :RejectionRate), Tuple{Int,Float64,Float64}}}(undef, N)

    k = 0

    for n in n_vec
        for ξ in ξ_vec
            k += 1

            model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
            rate = rejection_rate(SimpleScaling, model, template, n, simulation_size; tag_out=tag_out, q=q, level=level, seed = seed + k)

            results[k] = (n=Int(n), ξ=Float64(ξ), RejectionRate=rate)

            @info "Completed simulation" n ξ rate
        end
    end

    return DataFrame(results)
end

function run_simulation_generalscaling_errortype1(
    n_vec::AbstractVector{<:Integer},
    ξ_vec::AbstractVector{<:Real},
    simulation_size::Integer,
    template::IDFdata;
    tag_out = nothing,
    q::Integer=20,
    level::Real = .05,
    seed::Integer=1234)

    N = length(n_vec) * length(ξ_vec)

    results = Vector{NamedTuple{(:n, :ξ, :RejectionRate), Tuple{Int,Float64,Float64}}}(undef, N)

    k = 0

    for n in n_vec
        for ξ in ξ_vec
            k += 1

            model = GeneralScaling(d₀, μ₀, σ₀, ξ, α, δ)
            rate = rejection_rate(GeneralScaling, model, template, n, simulation_size; tag_out=tag_out, q=q, level=level, seed = seed + k)

            results[k] = (n=Int(n), ξ=Float64(ξ), RejectionRate=rate)

            @info "Completed simulation" n ξ rate
        end
    end

    return DataFrame(results)
end

# # Compilation warm-up - SimpleScaling
# model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
# template = rand(model, duration_dict, 1)
# rejection_rate(SimpleScaling, model, template, 50, 100, tag_out = "5min")

# run_simulation_simplescaling_errortype1(
#     [10],
#     [0.],
#     10,
#     template;
#     tag_out = "5min",
#     q = 20,
#     seed=1234)



# # Compilation warm-up - GeneralScaling
# model = GeneralScaling(d₀, μ₀, σ₀, ξ, α, δ)
# rejection_rate(GeneralScaling, model, template, 50, 100, tag_out = "5min")

# run_simulation_generalscaling_errortype1(
#     [10],
#     [0.],
#     10,
#     template;
#     tag_out = "5min",
#     q = 10,
#     seed=1234)


#  Simulation study

# results_simplescaling = run_simulation_simplescaling_errortype1(
#     n_vec,
#     ξ_vec,
#     simulation_size,
#     template;
#     tag_out = "5min",
#     q = 40,
#     seed=1234)

# CSV.write("SimpleScaling_type1_error.csv", results_simplescaling)

results_generalscaling = run_simulation_generalscaling_errortype1(
    n_vec,
    ξ_vec,
    simulation_size,
    template;
    tag_out = "5min",
    q = 20,
    seed=1234)

CSV.write("GeneralScaling_type1_error.csv", results_generalscaling)




## Troubleshooting

using Pkg
pkg"activate ."

using CSV, DataFrames, Distributions, IDFCurves, LinearAlgebra, Random
using Base.Threads

BLAS.set_num_threads(1)

# Simulation parameters
n_vec = [10, 15, 30, 50, 75, 100]
ξ_vec = collect(-0.4:0.2:0.4)
simulation_size = 10000


d₀ = 1.
μ₀ = 20
σ₀ = 5.
ξ = .05
α = .8
δ = .05

tags = ["5min", "10min", "15min", "30min", "1h", "2h", "6h", "12h", "24h"]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
duration_dict = Dict(zip(tags, durations))

model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
template = rand(model, duration_dict, 1)


rejection_rate(GeneralScaling,
    model,
    template,
    n,
    1000,
    tag_out = "5min",
    q = 20,
    level = 0.05,
    seed = 1234)




function rejection_rate(target_model::Type{<:MarginalScalingModel}, 
    model::MarginalScalingModel,
    template::IDFdata,
    sample_size::Integer,
    simulation_size::Integer;
    tag_out = nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234)

    if tag_out === nothing
        tag_out = IDFCurves._validation_tag(template)
    else
        tag_out = IDFCurves._validation_tag(template, tag_out)
    end

    duration_dict = getduration(template)

    # Using the generating model only as the optimizer starting point.
    if target_model === SimpleScaling
        initialmodel = IDFCurves.scalingtype(model)(duration(model), location(model), scale(model), 0., exponent(model))
    else
        initialmodel = IDFCurves.scalingtype(model)(duration(model), location(model), scale(model), 0., exponent(model), offset(model))
    end

    # One deterministic and independent seed per replication.
    seeds = rand(Xoshiro(seed), UInt64, simulation_size)

    # Do not use falses(M), which creates a bit-packed BitVector.
    reject = Vector{Bool}(undef, simulation_size)

    Threads.@threads :dynamic for m in eachindex(reject)
        rng = Xoshiro(seeds[m])

        data = rand(rng, model, duration_dict, sample_size)

        T = scalingtest(target_model, data, initialmodel; tag_out=tag_out, q=q)

        reject[m] = IDFCurves.pvalue(T) < level
    end

    return count(identity, reject) / simulation_size
end


