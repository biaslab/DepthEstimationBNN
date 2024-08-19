export TransformedDistribution

struct TransformedDistribution{D, F}
    dist::D
    f::F
end
@functor TransformedDistribution (dist, )

transform(d::TransformedDistribution) = reduce(∘, Iterators.reverse(d.f))(d.dist)
transform(d::TransformedDistribution{D,<:Function}) where { D } = d.f(d.dist)
transform(d::Distribution) = d