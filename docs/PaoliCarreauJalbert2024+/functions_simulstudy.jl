########################
### struct HybridScaling (only for generating data)
########################

struct HybridScaling <: MarginalScalingModel
    d₀::Real # reference duration
    d_break::Real # break duration
    μ₀::Real 
    σ₀::Real
    ξ::Real
    α_low::Real
    α_high::Real
end

Base.Broadcast.broadcastable(obj::HybridScaling) = Ref(obj)

duration(pd::HybridScaling) = pd.d₀
breakduration(pd::HybridScaling) = pd.d_break
exponent_low(pd::HybridScaling) = pd.α_low
exponent_high(pd::HybridScaling) = pd.α_high
location(pd::HybridScaling) = pd.μ₀
scale(pd::HybridScaling) = pd.σ₀
shape(pd::HybridScaling) = pd.ξ
params(pd::HybridScaling) = (location(pd), scale(pd), shape(pd), exponent_low(pd), exponent_high(pd))
params_number(::Type{<:HybridScaling}) = 5

function getdistribution(pd::HybridScaling, d::Real)
    
    μ₀ = location(pd)
    σ₀ = scale(pd)
    ξ = shape(pd)
    α_low = exponent_low(pd)
    α_high= exponent_high(pd)
    
    d₀ = duration(pd)
    d_break = breakduration(pd)

    if d_break <= d₀
        ls_dbreak = -α_high * (log(d_break) - log(d₀))
    else 
        ls_dbreak = -α_low * (log(d_break) - log(d₀))
    end
    s_dbreak = exp(ls_dbreak)
    μ_dbreak = μ₀ * s_dbreak
    σ_dbreak = σ₀ * s_dbreak
    if d >= d_break
        ls = -α_high * (log(d) - log(d_break))
        s = exp(ls)
        μ = μ_dbreak * s
        σ = σ_dbreak * s
    else 
        ls = -α_low * (log(d) - log(d_break))
        s = exp(ls)
        μ = μ_dbreak * s
        σ = σ_dbreak * s
    end
    
    return GeneralizedExtremeValue(μ, σ, ξ)
    
end


####################################################################################################################################

########################
### struct CompositeScaling (only for generating data)
########################

struct CompositeScaling <: MarginalScalingModel
    d₀::Real # reference duration
    μ₀::Real 
    σ₀::Real
    ξ::Real
    α_μ::Real
    α_σ::Real
end

Base.Broadcast.broadcastable(obj::CompositeScaling) = Ref(obj)

duration(pd::CompositeScaling) = pd.d₀
exponent_μ(pd::CompositeScaling) = pd.α_μ
exponent_σ(pd::CompositeScaling) = pd.α_σ
location(pd::CompositeScaling) = pd.μ₀
scale(pd::CompositeScaling) = pd.σ₀
shape(pd::CompositeScaling) = pd.ξ
params(pd::CompositeScaling) = (location(pd), scale(pd), shape(pd), exponent_μ(pd), exponent_σ(pd))
params_number(::Type{<:CompositeScaling}) = 5

function getdistribution(pd::CompositeScaling, d::Real)
    
    μ₀ = location(pd)
    σ₀ = scale(pd)
    ξ = shape(pd)
    α_μ = exponent_μ(pd)
    α_σ= exponent_σ(pd)
    
    d₀ = duration(pd)

    ls_μ = -α_μ * (log(d) - log(d₀))
    s_μ = exp(ls_μ)

    ls_σ = -α_σ * (log(d) - log(d₀))
    s_σ = exp(ls_σ)

    μ = μ₀ * s_μ
    σ = σ₀ * s_σ
    
    return GeneralizedExtremeValue(μ, σ, ξ)
    
end


####################################################################################################################################



function simul_test_results(generative_model::MarginalScalingModel, gen_durations::Vector{<:Real}, test_model::Type{<:MarginalScalingModel}, d_out::Real,
                                nyear::Int64, Nsimul::Int64)
    
    test_results = DataFrame(key = Int64[], pval = Float64[])
    
    for k in 1:Nsimul
        
        data = IDFCurves.rand(generative_model, gen_durations, nyear)

        try 
            global pval = IDFCurves.scalingtest(test_model, data, IDFCurves.gettag(data, d_out))
        catch e
            continue
        end

        push!(test_results, [k, pval])
        
    end

    return test_results
    
end

function get_bootstrap_interval(sample; β = 0.95, n_bootstrap = 10000)

    bs = bootstrap(mean, sample, BasicSampling(n_bootstrap))
    confint(bs, PercentileConfInt(β))
    
end

function get_boostrap_interval(data, data_column_name1, data_column_value1, data_column_name2, data_column_value2; 
                                α = 0.05, β = 0.95, n_bootstrap = 10000)

    symbol1, symbol2 = Symbol(data_column_name1), Symbol(data_column_name2)                            
    filtered_data = data[data[:,symbol1] .== data_column_value1 .&& data[:,symbol2] .== data_column_value2,:]
    sample =  filtered_data[:,:pval] .< α

    return get_bootstrap_interval(sample, β = β, n_bootstrap = n_bootstrap)[1]

end

function plot_rejection_rates(plot_data::DataFrame, range_ξ::AbstractArray, xaxis::String; 
                                xaxis_label = nothing,
                                xticks = :auto,
                                yticks = :auto,
                                log_scale_xaxis = false)

    layers = []
    for ξ in reverse(range_ξ) 
        data = plot_data[plot_data[:,:ξ] .== ξ, :]
        push!(layers, layer(data, x = Symbol(xaxis), y = :estim, color = :ξ, Geom.line()))
    end
    for ξ in reverse(range_ξ)
        data = plot_data[plot_data[:,:ξ] .== ξ, :]
        push!(layers, layer(data, x = Symbol(xaxis), ymin=:lower_bound, ymax=:upper_bound, color = :ξ, Geom.ribbon(), alpha = [0.4]))
    end

    palette = [Scale.color_continuous().f((2*length(range_ξ) - 2*i + 1)/(2*length(range_ξ))) for i in eachindex(range_ξ)]

    if isnothing(xaxis_label)
        if xaxis == "nyear"
            xaxis_label = "Length of the annual maxima series (n)"
        else 
            error("Please provide label for x-axis.")
        end
    end

    if log_scale_xaxis
        p = plot(layers..., Scale.x_log10(labels = x -> "$(round(10^x))"),
            Scale.color_discrete_manual(palette...),
            Guide.xlabel(xaxis_label),
            Guide.ylabel("Rejection rate"),
            Guide.xticks(ticks = xticks),
            Guide.yticks(ticks = yticks),
            style(major_label_font_size = 12pt, minor_label_font_size = 10pt,
                key_title_font_size = 12pt, key_label_font_size = 10pt)
        )
    else 
        p = plot(layers...,
            Scale.color_discrete_manual(palette...),
            Guide.xlabel(xaxis_label),
            Guide.ylabel("Rejection rate"),
            Guide.xticks(ticks = xticks),
            Guide.yticks(ticks = yticks),
            style(major_label_font_size = 12pt, minor_label_font_size = 10pt,
                key_title_font_size = 12pt, key_label_font_size = 10pt)
        )
    end

    return p

end