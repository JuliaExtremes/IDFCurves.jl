
abstract type EllipticalCopula end

function getcormatrix(obj::EllipticalCopula)
    return obj.cormatrix
end

## GaussianCopula

struct GaussianCopula <: EllipticalCopula
    cormatrix::PDMat
     
    GaussianCopula(Σ::AbstractMatrix{<:Real}) = new(PDMat(Σ))

end


function logpdf(C::GaussianCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    D = MvNormal(getcormatrix(C))
    
    x = quantile.(Normal(), u)

    return logpdf(D, x) - sum(logpdf.(Normal(), x))

end

## TCopula

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
