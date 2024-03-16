struct TCopula{df} <: EllipticalCopula
    cormatrix::PDMat

    function TCopula{df}(cormatrix::AbstractMatrix{<:Real}) where df
        @assert (df isa Integer) "The given number of degrees of freedom must be an integer for the Student copula"
        new(PDMat(cormatrix))
    end

end

function TCopula(df::Integer, cormatrix::AbstractMatrix{<:Real})
    TCopula{df}(cormatrix)
end

#TODO Customize error message when TCopula(::AbstractMatrix{<:Real}) is called without specifying df.

function dof(C::TCopula{df}) where df
    return df
end

function dof(Ctype::Type{TCopula{df}}) where df
    return df
end

function getcormatrix(C::TCopula)
    return C.cormatrix
end

function logpdf(C::TCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    ν = dof(C)
    D = MvTDist(ν, getcormatrix(C))
    
    x = quantile.(TDist(ν), u)

    return logpdf(D, x) - sum(logpdf.(TDist(ν), x))

end