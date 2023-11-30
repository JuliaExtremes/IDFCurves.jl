# Function from ExtendedExtremes.jl
"""
    qqplot(y::Vector{Float64}, pd::Distribution)

Quantile-Quantile plot
"""
function qqplot(pd::Distribution, y::Vector{Float64})

    n = length(y)
    q = sort(y)

    p = (1:n) ./ (n+1)

    q̂ = quantile.(pd, p);
    
    l1 = layer(y = q, x = q̂, Geom.point)
    l2 = layer(y = q[[1, end]], x = q[[1, end]], Geom.line, Theme(default_color="red", line_style=[:dash]))

    return plot(l2, l1,
        Guide.xlabel("Model"), Guide.ylabel("Empirical"), Guide.title("Quantile Plot"),
        Theme(discrete_highlight_color=c->nothing))
end