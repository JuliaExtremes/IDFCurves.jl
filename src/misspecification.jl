
"""
    godambe(fd::Distribution, y::AbstractVector{<:Real})

Estimation the Godambe information matrix of the model `fd` according to the data `y`.

## Reference

See also [`variability_matrix`](@ref).

Varin, C., Reid, N., & Firth, D. (2011). An overview of composite likelihood methods. *Statistica Sinica*, 21 (1), 5–42.

"""
function godambe(fd::Distribution, y::AbstractVector{<:Real})
   
    J = variability_matrix(fd, y)
    H = hessian(fd, y)
    
    G = PDMats.X_invA_Xt(J, H)
    
end


"""
    hessian(fd::Distribution, y::AbstractVector{<:Real})

Estimation the sensitivity matrix of the model `fd` according to the data `y`.

## Reference

See also [`godambe`](@ref) and [`variability_matrix`](@ref).

Varin, C., Reid, N., & Firth, D. (2011). An overview of composite likelihood methods. *Statistica Sinica*, 21 (1), 5–42.

"""
function hessian(fd::Distribution, y::AbstractVector{<:Real})
   
    θ̂ = collect(params(fd))
    
    dist = eval(nameof(typeof(fd)))
    
    ll(θ::AbstractVector{<:Real}) = -sum(logpdf.(dist(θ...), y))
        
    H = ForwardDiff.hessian(ll, θ̂)
    
    return PDMat(Symmetric(H))
    
end


"""
    variability_matrix(fd::Distribution, y::AbstractVector{<:Real})

Estimate the variability matrix of model `fd` according to the data `y`.

## Details

See also [`godambe`](@ref).

Varin, C., Reid, N., & Firth, D. (2011). An overview of composite likelihood methods. Statistica Sinica, 21 (1), 5–42.
"""
function variability_matrix(fd::Distribution, y::AbstractVector{<:Real})
    
    θ̂ = collect(params(fd))
    
    dist = eval(nameof(typeof(fd)))
    
    pd(θ::AbstractVector{<:Real}) = dist(θ...)
    ll(θ, y) = sum(logpdf.(pd(θ), y))
    u(θ, y) = ForwardDiff.gradient( θ -> ll(θ, y), θ)
    
    J = zeros(length(θ̂), length(θ̂))
    
    for yᵢ in y
       
        uᵢ = u(θ̂, yᵢ)
        
        J .+= uᵢ * uᵢ'
        
    end
    
    return PDMat(Symmetric(J))
    
end