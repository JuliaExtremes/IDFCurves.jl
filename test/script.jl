using IDFCurves, Test

using CSV, DataFrames, Distributions, Extremes, Gadfly, LinearAlgebra, SpecialFunctions, Optim, ForwardDiff, BesselK, SpecialFunctions, PDMats

df = CSV.read(joinpath("data","702S006.csv"), DataFrame)
    
tags = names(df)[2:10]
durations = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
h = IDFCurves.logdist(durations)
duration_dict = Dict(zip(tags, durations))
    
data = IDFdata(df, "Year", duration_dict)



# Redéfinition des méthodes de fitting pour les DependentScalingModels :

abstract_model = DependentScalingModel{SimpleScaling, MaternCorrelationStructure, TCopula}
model = IDFCurves.construct_model(abstract_model, data, 1, [IDFCurves.map_to_real_space(SimpleScaling, [20, 5, .04, .76]); [0., 0.]])

# Copule de Student :

struct StudentCopula{df} <: EllipticalCopula
    cormatrix::PDMat

    function StudentCopula{df}(cormatrix::AbstractMatrix{<:Real}) where df
        @assert df isa Integer
        new(PDMat(cormatrix))
    end

end

function dof(C::StudentCopula{df}) where df
    return df
end

function dof(Ctype::Type{StudentCopula{df}}) where df
    return df
end


cop = StudentCopula{12}([1 0; 0 1])
cop_type = StudentCopula{7}
dof(cop) # renvoie 12
dof(cop_type) # renvoie 7
dof(StudentCopula) # renvoie une erreur

cop_type([1 0; 0 1])
# Tests sur l'initialisation :

fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, .04, .76]) # renvoie résultats
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 2*eps(), .76]) # renvoie erreur
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 1000*eps(), .76]) # renvoie résultat où ξ=0
fm = IDFCurves.fit_mle(SimpleScaling, data, 1, [20, 5, 0.0001, .76]) # renvoie même résultat que le premier
