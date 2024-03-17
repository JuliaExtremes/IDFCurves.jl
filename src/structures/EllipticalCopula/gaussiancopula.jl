struct GaussianCopula <: EllipticalCopula
    cormatrix::PDMat
    GaussianCopula(Σ::AbstractMatrix{<:Real}) = new(PDMat(Σ))
end

function getcormatrix(C::GaussianCopula)
    return C.cormatrix
end

function logpdf(C::GaussianCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    D = MvNormal(getcormatrix(C))
    
    x = quantile.(Normal(), u)

    return logpdf(D, x) - sum(logpdf.(Normal(), x))

end