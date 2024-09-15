struct IdentityCopula <: EllipticalCopula

end

IdentityCopula(Σ::Any) = IdentityCopula()

function logpdf(C::IdentityCopula, u::AbstractVector{<:Real})
    @assert all(0 .≤ u .≤ 1) 
    
    return 0

end

function logpdf(C::IdentityCopula, u::AbstractArray{<:AbstractArray{<:Real}})
    @assert all(0 .≤ vcat(u...) .≤ 1) 
    
    return 0

end