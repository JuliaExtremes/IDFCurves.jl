struct TCopula <: EllipticalCopula
    df::Real
    cormatrix::PDMat
     
    TCopula(df::Real, Σ::AbstractMatrix{<:Real}) = new(df, PDMat(Σ))

end

function getcormatrix(obj::TCopula)
    return obj.cormatrix
end

function dof(C::TCopula)
    return C.df
end

function logpdf(C::TCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    ν = dof(C)
    D = MvTDist(ν, getcormatrix(C))
    
    x = quantile.(TDist(ν), u)

    return logpdf(D, x) - sum(logpdf.(TDist(ν), x))

end