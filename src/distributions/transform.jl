export TransformedDistribution

struct TransformedDistribution{D, F}
    dist::D
    f::F
end
@functor TransformedDistribution (dist, )

transform(d::TransformedDistribution) = _transform(d)
transform(d::TransformedDistribution{D,<:Function}) where { D } = d.f(d.dist)
transform(d::Distribution) = d

@generated function _transform(d::TransformedDistribution{D,<:NTuple{N, Any}}) where {D,N}

    # create symbols
    symbols = vcat(:(d.dist), [gensym() for _ in 1:N])

    # create calls
    calls = [:( $(symbols[i+1]) = d.f[$i]($(symbols[i])) ) for i in 1:N]

    return Expr(:block, calls...)
end