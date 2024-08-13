export Chain

struct Chain{T<:Union{Tuple, NamedTuple, AbstractVector}}
    layers :: T
end
@functor Chain

Chain(xs...) = Chain(xs)

(c::Chain)(x_mean, x_var) = _applychain(c.layers, x_mean, x_var)
(c::Chain)(x) = _applychain(c.layers, x)

@generated function _applychain(layers::Tuple{Vararg{Any,N}}, x_mean, x_var) where {N}
    symbols = vcat(:x_mean, :x_var, [gensym() for _ in 1:2*N])
    calls = [:( ($(symbols[2*i+1]), $(symbols[2*i+2])) = layers[$i]($(symbols[2*i-1]), $(symbols[2*i]))) for i in 1:N]
    return Expr(:block, calls...)
end
@generated function _applychain(layers::Tuple{Vararg{Any,N}}, x) where {N}
    symbols = vcat(:x, [gensym() for _ in 1:N])
    calls = [:( $(symbols[i+1]) = layers[$i]($(symbols[i])) ) for i in 1:N]
    return Expr(:block, calls...)
end

KL_loss(c::Chain) = mapreduce(l -> KL_loss(l), +, c.layers)