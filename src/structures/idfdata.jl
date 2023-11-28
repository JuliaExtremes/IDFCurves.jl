
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


gettag(s::IDFdata) = s.tag



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