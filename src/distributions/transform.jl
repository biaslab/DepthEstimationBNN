export TransformedDistribution

struct TransformedDistribution{D, F}
    dist::D
    f::F
end
@functor TransformedDistribution (dist, )

transform(d::TransformedDistribution) = reduce(∘, d.f)(d.dist)
transform(d::Distribution) = d