
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

function rejection_rate(
    target_model::Type{<:MarginalScalingModel},
    model::MarginalScalingModel,
    template::IDFdata,
    sample_size::Integer,
    simulation_size::Integer;
    tag_out=nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234,
)

    sample_size > 0 ||
        throw(ArgumentError("sample_size must be positive."))

    simulation_size > 0 ||
        throw(ArgumentError("simulation_size must be positive."))

    q > 0 ||
        throw(ArgumentError("q must be positive."))

    0 < level < 1 ||
        throw(ArgumentError("level must lie in (0, 1), got level=$level."))

    scalingtype(model) === target_model ||
        throw(ArgumentError("For a Type I error study, the generating and fitted models must have the same type, got $(scalingtype(model)) and $target_model."))

    tag_out = if isnothing(tag_out)
        IDFCurves._validation_tag(template)
    else
        IDFCurves._validation_tag(template, tag_out)
    end

    duration_dict = getduration(template)

    # Use the generating parameters as starting values, except for ξ = 0.
    initialmodel = IDFCurves._neutral_initial_model(model)

    # One deterministic and independent seed per replication.
    seeds = rand(Xoshiro(seed), UInt64, simulation_size)

    # Vector{Bool} is preferable to BitVector for threaded writes.
    rejected = Vector{Bool}(undef, simulation_size)
    valid = Vector{Bool}(undef, simulation_size)

    Threads.@threads :dynamic for m in eachindex(rejected)
        rng = Xoshiro(seeds[m])

        data = rand(rng, model, duration_dict, sample_size)

        test = scalingtest(target_model, data, initialmodel; tag_out=tag_out, q=q)

        valid[m] = IDFCurves.isvalid(test)

        rejected[m] = if valid[m]
            IDFCurves.decision(test, level)
        else
            false
        end
    end

    nvalid = count(valid)
    nfailed = simulation_size - nvalid

    nvalid > 0 || 
        throw(ErrorException("None of the $simulation_size maximum likelihood fits converged."))

    nrejected = count(rejected)
    rate = nrejected / nvalid

    return (
        rejection_rate=rate,
        nrejected=nrejected,
        nvalid=nvalid,
        nfailed=nfailed,
    )
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
            res = rejection_rate(SimpleScaling, model, template, n, simulation_size; tag_out=tag_out, q=q, level=level, seed = seed + k)

            results[k] = (n=Int(n), ξ=Float64(ξ), RejectionRate=res.rejection_rate)

            @info "Completed simulation" n ξ res.rejection_rate 
            @info "Number of discarded simulations : " res.nfailed " / " res.nvalid+res.nfailed
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
            res = rejection_rate(GeneralScaling, model, template, n, simulation_size; tag_out=tag_out, q=q, level=level, seed = seed + k)

            results[k] = (n=Int(n), ξ=Float64(ξ), RejectionRate=res.rejection_rate)

            @info "Completed simulation" n ξ res.rejection_rate 
            @info "Number of discarded simulations : " res.nfailed " / " res.nvalid+res.nfailed
        end
    end

    return DataFrame(results)
end

# Compilation warm-up - SimpleScaling
model = SimpleScaling(d₀, μ₀, σ₀, ξ, α)
template = rand(model, duration_dict, 1)
rejection_rate(SimpleScaling, model, template, 50, 100, tag_out = "5min")

run_simulation_simplescaling_errortype1(
    [10],
    [0.],
    10,
    template;
    tag_out = "5min",
    q = 20,
    seed=1234)






#  Simulation study

results_simplescaling = run_simulation_simplescaling_errortype1(
    n_vec,
    ξ_vec,
    simulation_size,
    template;
    tag_out = "5min",
    q = 40,
    seed=1234)

CSV.write("SimpleScaling_type1_error.csv", results_simplescaling)

results_generalscaling = run_simulation_generalscaling_errortype1(
    n_vec,
    ξ_vec,
    simulation_size,
    template;
    tag_out = "5min",
    q = 20,
    seed=1234)

CSV.write("GeneralScaling_type1_error.csv", results_generalscaling)
