abstract type Distribution end

include("distributions/normal.jl")
include("distributions/transform.jl")
include("distributions/truncate.jl")
include("distributions/discrete.jl")

function Base.:*(d::Distribution, h::ShiftedHeaviside)
    lower, upper = support(d)
    return truncate(d, max(h.a, lower), min(Inf, upper))
end
Base.:*(h::ShiftedHeaviside, d::Distribution) = d * h

struct SupportDistribution{D <: Distribution, L, U } <: Distribution
    dist::D
    SupportDistribution(d::D) where { D <: Distribution } = new{D, support(d)...}(d)
end
@functor SupportDistribution