abstract type Distribution end

include("distributions/truncate.jl")
include("distributions/discrete.jl")
include("distributions/normal.jl")
include("distributions/poisson.jl")
include("distributions/transform.jl")

struct SupportDistribution{D <: Distribution, L, U } <: Distribution
    dist::D
    function SupportDistribution(d::D) where { D <: Distribution } 
        s = support(d)
        return new{D, first(s), last(s)}(d)
    end
end
@functor SupportDistribution

pmf(::Distribution, ::Real) = 0
pdf(::Distribution, ::Real) = 0