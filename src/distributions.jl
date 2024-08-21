abstract type Distribution end

include("distributions/discrete.jl")
include("distributions/normal.jl")
include("distributions/poisson.jl")
include("distributions/transform.jl")
include("distributions/truncate.jl")
include("distributions/truncated_poisson.jl")

function Base.:*(d::Distribution, h::ShiftedHeaviside)
    lower, upper = support(d)
    return truncate(d, max(h.a, lower), min(Inf, upper))
end
Base.:*(h::ShiftedHeaviside, d::Distribution) = d * h

struct SupportDistribution{D <: Distribution, L, U } <: Distribution
    dist::D
    function SupportDistribution(d::D) where { D <: Distribution } 
        s = support(d)
        return new{D, first(s), last(s)}(d)
    end
end
@functor SupportDistribution