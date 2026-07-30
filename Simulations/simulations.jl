# julia --project=. --threads=auto Simulations/simulations.jl

using CSV
using DataFrames
using Distributions
using IDFCurves
using LinearAlgebra
using Random
using Base.Threads

include(joinpath(@__DIR__, "hybridscaling.jl"))

BLAS.set_num_threads(1)

# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------

const D₀ = 1.0
const Μ₀ = 20.0
const Σ₀ = 4.0
const Α = 0.8
const Δ = 0.05

const TAGS = [
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

const DURATIONS = [
    1 / 12,
    1 / 6,
    1 / 4,
    1 / 2,
    1.0,
    2.0,
    6.0,
    12.0,
    24.0,
]

const DURATION_DICT = Dict(zip(TAGS, DURATIONS))

const N_VEC = [10, 15, 30, 50, 75, 100]
const Ξ_VEC = collect(-0.4:0.2:0.4)
const Δ_VEC = collect(0.0:0.01:0.05)
const A_VEC = collect(.4:.05:.8)

const SIMULATION_SIZE = 15000
const LEVEL = 0.05

const OUTPUT_DIR = joinpath(@__DIR__, "simulation_results")
mkpath(OUTPUT_DIR)

const TEMPLATE_MODEL = SimpleScaling(D₀, Μ₀, Σ₀, 0.05, Α)
const TEMPLATE = rand(Xoshiro(1234), TEMPLATE_MODEL, DURATION_DICT, 1)

# -----------------------------------------------------------------------------
# Core simulation function
# -----------------------------------------------------------------------------

"""
    rejection_rate(
        target_model,
        generating_model,
        template,
        sample_size,
        simulation_size;
        tag_out=nothing,
        q=20,
        level=0.05,
        seed=1234,
    )

Estimate the rejection rate of the training-validation Cramér--von Mises test.

Samples are generated independently from `generating_model`, while
`target_model` specifies the scaling model fitted by `scalingtest`. Simulations
for which maximum likelihood estimation does not converge are excluded from the
rejection-rate calculation and reported through `nfailed`.
"""
function rejection_rate(
    target_model::Type{<:MarginalScalingModel},
    generating_model::MarginalScalingModel,
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
        throw(ArgumentError(
            "level must lie in (0, 1), got level=$level.",
        ))

    tag_out = if isnothing(tag_out)
        IDFCurves._validation_tag(template)
    else
        IDFCurves._validation_tag(template, tag_out)
    end

    duration_dict = getduration(template)

    # One deterministic and independent RNG seed per replication.
    seeds = rand(Xoshiro(seed), UInt64, simulation_size)

    # Vector{Bool}, rather than BitVector, is used for independent threaded writes.
    rejected = zeros(Bool, simulation_size)
    valid = zeros(Bool, simulation_size)

    Threads.@threads :dynamic for m in eachindex(rejected)
        rng = Xoshiro(seeds[m])

        data = rand(
            rng,
            generating_model,
            duration_dict,
            sample_size,
        )

        # The fitted model is initialized automatically from the simulated
        # training sample. This works both under the null and under alternatives
        # generated from a different scaling-model family.
        try
            test = scalingtest(
                target_model,
                data;
                tag_out=tag_out,
                q=q,
            )

            valid[m] = IDFCurves.isvalid(test)
            rejected[m] =
                valid[m] && IDFCurves.decision(test, level)
        catch error
            valid[m] = false
        end
    end

    nvalid = count(valid)
    nfailed = simulation_size - nvalid

    nvalid > 0 ||
        throw(ErrorException(
            "No simulation produced a valid test.",
        ))

    nrejected = count(rejected)

    return (
        rejection_rate=nrejected / nvalid,
        nrejected=nrejected,
        nvalid=nvalid,
        nfailed=nfailed,
    )
end

# -----------------------------------------------------------------------------
# Type I error simulations
# -----------------------------------------------------------------------------

function run_simulation_simplescaling_type1(
    n_vec::AbstractVector{<:Integer},
    ξ_vec::AbstractVector{<:Real},
    simulation_size::Integer,
    template::IDFdata;
    tag_out=nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234,
)
    N = length(n_vec) * length(ξ_vec)

    results = Vector{
        NamedTuple{
            (:n, :ξ, :RejectionRate, :NRejected, :NValid, :NFailed),
            Tuple{Int,Float64,Float64,Int,Int,Int},
        },
    }(undef, N)

    k = 0

    for n in n_vec
        for ξ in ξ_vec
            k += 1

            generating_model =
                SimpleScaling(D₀, Μ₀, Σ₀, ξ, Α)

            res = rejection_rate(
                SimpleScaling,
                generating_model,
                template,
                n,
                simulation_size;
                tag_out=tag_out,
                q=q,
                level=level,
                seed=seed + k,
            )

            results[k] = (
                n=Int(n),
                ξ=Float64(ξ),
                RejectionRate=res.rejection_rate,
                NRejected=res.nrejected,
                NValid=res.nvalid,
                NFailed=res.nfailed,
            )

            total = res.nvalid + res.nfailed

            @info(
                "Completed SimpleScaling Type I simulation",
                n=n,
                ξ=ξ,
                rejection_rate=res.rejection_rate,
                discarded=res.nfailed,
                total=total,
            )
        end
    end

    return DataFrame(results)
end

function run_simulation_generalscaling_type1(
    n_vec::AbstractVector{<:Integer},
    ξ_vec::AbstractVector{<:Real},
    simulation_size::Integer,
    template::IDFdata;
    tag_out=nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234,
)
    N = length(n_vec) * length(ξ_vec)

    results = Vector{
        NamedTuple{
            (:n, :ξ, :RejectionRate, :NRejected, :NValid, :NFailed),
            Tuple{Int,Float64,Float64,Int,Int,Int},
        },
    }(undef, N)

    k = 0

    for n in n_vec
        for ξ in ξ_vec
            k += 1

            generating_model =
                GeneralScaling(D₀, Μ₀, Σ₀, ξ, Α, Δ)

            res = rejection_rate(
                GeneralScaling,
                generating_model,
                template,
                n,
                simulation_size;
                tag_out=tag_out,
                q=q,
                level=level,
                seed=seed + k,
            )

            results[k] = (
                n=Int(n),
                ξ=Float64(ξ),
                RejectionRate=res.rejection_rate,
                NRejected=res.nrejected,
                NValid=res.nvalid,
                NFailed=res.nfailed,
            )

            total = res.nvalid + res.nfailed

            @info(
                "Completed GeneralScaling Type I simulation",
                n=n,
                ξ=ξ,
                rejection_rate=res.rejection_rate,
                discarded=res.nfailed,
                total=total,
            )
        end
    end

    return DataFrame(results)
end

# -----------------------------------------------------------------------------
# Power simulations
# -----------------------------------------------------------------------------

function run_simulation_simplescaling_power(
    δ_vec::AbstractVector{<:Real},
    ξ_vec::AbstractVector{<:Real},
    simulation_size::Integer,
    template::IDFdata;
    sample_size::Integer=60,
    tag_out=nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234,
)
    N = length(δ_vec) * length(ξ_vec)

    results = Vector{
        NamedTuple{
            (:δ, :ξ, :RejectionRate, :NRejected, :NValid, :NFailed),
            Tuple{Float64,Float64,Float64,Int,Int,Int},
        },
    }(undef, N)

    k = 0

    for δ in δ_vec
        for ξ in ξ_vec
            k += 1

            # GeneralScaling with δ = 0 corresponds to the SimpleScaling null.
            generating_model =
                GeneralScaling(D₀, Μ₀, Σ₀, ξ, Α, δ)

            res = rejection_rate(
                SimpleScaling,
                generating_model,
                template,
                sample_size,
                simulation_size;
                tag_out=tag_out,
                q=q,
                level=level,
                seed=seed + k,
            )

            results[k] = (
                δ=Float64(δ),
                ξ=Float64(ξ),
                RejectionRate=res.rejection_rate,
                NRejected=res.nrejected,
                NValid=res.nvalid,
                NFailed=res.nfailed,
            )

            total = res.nvalid + res.nfailed

            @info(
                "Completed SimpleScaling power simulation",
                n=sample_size,
                δ=δ,
                ξ=ξ,
                rejection_rate=res.rejection_rate,
                discarded=res.nfailed,
                total=total,
            )
        end
    end

    return DataFrame(results)
end

function run_simulation_generalscaling_power(
    α_vec::AbstractVector{<:Real},
    ξ_vec::AbstractVector{<:Real},
    simulation_size::Integer,
    template::IDFdata;
    sample_size::Integer=60,
    tag_out=nothing,
    q::Integer=20,
    level::Real=0.05,
    seed::Integer=1234,
)

    N = length(α_vec) * length(ξ_vec)

    results = Vector{
        NamedTuple{
            (:α₁, :ξ, :RejectionRate, :NRejected, :NValid, :NFailed),
            Tuple{Float64,Float64,Float64,Int,Int,Int},
        },
    }(undef, N)

    k = 0

    for α₁ in α_vec
        for ξ in ξ_vec
            k += 1

            # At α₁ = α₂ and HybridScaling reduces to simple scaling.
            generating_model =
                HybridScaling(D₀, Μ₀, Σ₀, ξ, α₁, Α)

            res = rejection_rate(
                GeneralScaling,
                generating_model,
                template,
                sample_size,
                simulation_size;
                tag_out=tag_out,
                q=q,
                level=level,
                seed=seed + k,
            )

            results[k] = (
                α₁=Float64(α₁),
                ξ=Float64(ξ),
                RejectionRate=res.rejection_rate,
                NRejected=res.nrejected,
                NValid=res.nvalid,
                NFailed=res.nfailed,
            )

            total = res.nvalid + res.nfailed

            @info(
                "Completed GeneralScaling power simulation",
                n=sample_size,
                α₁=α₁,
                ξ=ξ,
                rejection_rate=res.rejection_rate,
                discarded=res.nfailed,
                total=total,
            )
        end
    end

    return DataFrame(results)
end

# -----------------------------------------------------------------------------
# Run simulation study
# -----------------------------------------------------------------------------

results_simplescaling_type1 = run_simulation_simplescaling_type1(
    N_VEC,
    Ξ_VEC,
    SIMULATION_SIZE,
    TEMPLATE;
    tag_out="5min",
    q=40,
    level=LEVEL,
    seed=1234,
)

CSV.write(
    joinpath(OUTPUT_DIR, "SimpleScaling_type1_error.csv"),
    results_simplescaling_type1,
)

results_generalscaling_type1 = run_simulation_generalscaling_type1(
    N_VEC,
    Ξ_VEC,
    SIMULATION_SIZE,
    TEMPLATE;
    tag_out="5min",
    q=40,
    level=LEVEL,
    seed=1234,
)

CSV.write(
    joinpath(OUTPUT_DIR, "GeneralScaling_type1_error.csv"),
    results_generalscaling_type1,
)

results_simplescaling_power = run_simulation_simplescaling_power(
    Δ_VEC,
    Ξ_VEC,
    SIMULATION_SIZE,
    TEMPLATE;
    sample_size=60,
    tag_out="5min",
    q=40,
    level=LEVEL,
    seed=1234,
)

CSV.write(
    joinpath(OUTPUT_DIR, "SimpleScaling_power.csv"),
    results_simplescaling_power,
)

results_generalscaling_power = run_simulation_generalscaling_power(
    A_VEC,
    Ξ_VEC,
    SIMULATION_SIZE,
    TEMPLATE;
    sample_size=60,
    tag_out="5min",
    q=40,
    level=LEVEL,
    seed=1234,
)

CSV.write(
    joinpath(OUTPUT_DIR, "GeneralScaling_power.csv"),
    results_generalscaling_power,
)
