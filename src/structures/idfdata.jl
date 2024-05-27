
struct IDFdata
    tag::Vector{<:AbstractString}
    duration::Dict{String, T} where T<:Real
    year::Dict{String, Vector{T}} where T<:Int
    data::Dict{String, Vector{T}} where T<:Real
    
    function IDFdata(tag::Vector{<:AbstractString},
        duration::Dict{String, T} where T<:Real, 
        year::Dict{String, Vector{T}} where T<:Int,
        data::Dict{String, Vector{T}} where T<:Real)
        
        @assert issetequal(keys(duration), tag) "Tags must match the dictionary keys containing the durations"
        @assert issetequal(keys(year), tag) "Tags must match the dictionary keys containing the years"
        @assert issetequal(keys(data), tag) "Tags must match the dictionary keys containing the data"
        
        return new(tag, duration, year, data)
        
    end
end

"""
    IDFdata(df::DataFrame, year_id::String, duration::Dict{String, T} where T<:Real)

Construct a IDFdata structure from a DataFrame.

### Details

- `year_id`: string indicating the year column
- `duration`: Dictionary mapping the dataframe id to duration.

See the tutorial for an example.
"""
function IDFdata(df::DataFrame, year_id::String, duration::Dict{String, T} where T<:Real)
    x = df[:, year_id]
    
    year = Dict{String, Vector{Int64}}()
    data = Dict{String, Vector{Float64}}()

    tags = collect(keys(duration))
    
    for k in tags
    
        yₖ = df[:, k]

        id = ismissing.(yₖ)

        year[k] = x[.!(id)]
        data[k] = yₖ[.!(id)]

    end
    
    return IDFdata(tags, duration, year, data)
    
end

Base.Broadcast.broadcastable(obj::IDFdata) = Ref(obj)

"""
    getdata(s::IDFdata)

Return a dictionary containing the data vector for each duration tag.
"""
getdata(s::IDFdata) = s.data

"""
    getdata(s::IDFdata, tag::String)

Return the data vector corresponding to the duration tag `tag`.
"""
function getdata(s::IDFdata, tag::String)
    @assert tag in keys(getdata(s)) "The specified duration is not in the dataset."
        
    return getdata(s)[tag]
end

"""
    getdata(s::IDFdata, tag::String, year::Int)

Return the data corresponding to the duration tag `tag` and the year `year`.
"""
function getdata(s::IDFdata, tag::String, year::Int)
    @assert tag in keys(getyear(s)) "The specified duration is not in the dataset."
    @assert year in getyear(s, tag) "The specified year is not in the dataset."
   
    id = findfirst(getyear(s, tag) .== year)
    
    v = getdata(s)[tag]
    
    return v[id]
    
end

"""
    getduration(s::IDFdata)

Return a dictionary containing the duration vector for each duration tag.
"""
getduration(s::IDFdata) = s.duration

"""
    getduration(s::IDFdata, tag::String)

Return the vector of durations for the duration tag `tag`.
"""
function getduration(s::IDFdata, tag::String)
    @assert tag in keys(getduration(s)) "The specified duration is not in the dataset."
    return getduration(s)[tag]
end

"""
    getyear(s::IDFdata)

Return a dictionary containing the year vector for each duration tag.
"""
getyear(s::IDFdata) = s.year

"""
    getyear(s::IDFdata, tag::String)

Return the vector of years for the duration tag `tag`.
"""
function getyear(s::IDFdata, tag::String)
    @assert tag in keys(getyear(s)) "The specified duration is not in the dataset."
    return getyear(s)[tag]
end

"""
    gettag(data::IDFdata)

Return the tag list.
"""
function gettag(data::IDFdata)

    k = collect(keys(getduration(data)))
    v = collect(values(getduration(data)))

    ind = sortperm(v)

    return k[ind]
end

"""
    gettag(data::IDFdata, d::Real)

Return the tag corresponding to the duration `d` if it exists; throw an error otherwise.
"""
function gettag(data::IDFdata, d::Real)

    duration_dict = getduration(data)

    has(x::Real, y::Real) = x ≈ y
    has(x::Vector{<:Real}, y::Real) = any(has(i, y) for i in x)

    key_value = [k for (k,v) in duration_dict if has(v, d)] 

    if isempty(key_value)
        error("the specified duration does not correspond to a tag.")
    else
        return key_value[]
    end

end

"""
    excludeduration(data::IDFdata, d::Real)

Remove the data of `data` corresponding to the duration `d`.
"""
function excludeduration(data::IDFdata, d::Real)

    new_year = Dict{String, Vector{Int64}}()
    new_data = Dict{String, Vector{Float64}}()
    new_duration = Dict(k => v for (k, v) in getduration(data) if k != gettag(data, d))

    new_tag = collect(keys(new_duration))
    for key in new_tag
        new_year[key] = getyear(data, key)
        new_data[key] = getdata(data, key)
    end

    return IDFdata(new_tag, new_duration, new_year, new_data)

end

"""
    Base.show(io::IO, obj::IDFdata)

Override of the show function for the objects of type IDFdata.

"""
function Base.show(io::IO, obj::IDFdata)
    prefix = "  "
    println(io, "IDFdata")
    for tag in gettag(obj)
        println(io, prefix, tag, ": ", typeof(getdata(obj, tag)), "[", length(getdata(obj,tag)), "]" )
    end
end


"""
    getKendalldata(obj::IDFdata)

Computes the Kendall tau for each pair of durations for which obj contains data,
    and returns them in a DataFrame.

"""
function getKendalldata(obj::IDFdata)

    df_kendall = DataFrame(tag1 = String[], tag2 = String[], distance = Float64[], kendall = Float64[])
    
    for c in combinations(gettag(obj),2)
    
        d₁ = getduration(obj, c[1])
        d₂ = getduration(obj, c[2])
        
        h = IDFCurves.logdist(d₁, d₂)
        
        y₁ = getdata(obj, c[1])
        y₂ = getdata(obj, c[2])
        
        τ = corkendall(y₁, y₂)
        
        push!(df_kendall, [c[1], c[2], h, τ])
        
    end

    return df_kendall

end
