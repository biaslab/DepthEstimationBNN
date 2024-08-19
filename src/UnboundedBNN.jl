module UnboundedBNN

using StatsFuns, Functors, LogExpFunctions

export Unbounded

include("math.jl")

include("losses/KL.jl")

include("layers/chain.jl")
include("layers/skip.jl")
include("layers/linear.jl")
include("layers/linear_bbb.jl")
include("layers/linear_spike.jl")
include("layers/relu.jl")
include("layers/leaky_relu.jl")
include("layers/softmax.jl")

include("distributions.jl")

include("data/spiral.jl")

struct Unbounded{T1, T2, T3, P, TP}
    input_layer::T1
    hidden_layers::T2
    output_layers::T3
    prior::P
    posterior::TP
end
@functor Unbounded (input_layer, hidden_layers, output_layers, posterior) # exclude prior from training

(m::Unbounded)(x) = _applymodel(m, SupportDistribution(transform(m.posterior)), x)

@generated function _applymodel(model::Unbounded, post::SupportDistribution{D,MIN,MAX}, x) where {D, MIN, MAX}

    # create symbols
    state_symbols = vcat(:x, [gensym() for _ in 1:(MAX + 1)])
    output_symbols = [gensym() for _ in 1:(MAX - MIN + 1)]

    # create calls
    expand_call = :( $(state_symbols[2]) = model.input_layer($(state_symbols[1])) )
    state_calls = [:( $(state_symbols[i+2]) = model.hidden_layers[$i]($(state_symbols[i+1])) ) for i in 1:MAX]
    output_calls = [:( $(output_symbols[i]) = model.output_layers[$(MIN+i)]($(state_symbols[MIN + i + 1])) ) for i in 1:(MAX - MIN + 1)]
    return_call = :( return $(join_as_tuples(output_symbols) ) )
    calls = vcat(expand_call, state_calls, output_calls, return_call)

    return Expr(:block, calls...)

end

# @generated function _applymodel(model::Unbounded, post::SafeTruncatedPoisson{MIN, MAX, T}, x_mean, x_var) where {MIN, MAX, T}

#     # create symbols
#     state_symbols = vcat(:x_mean, :x_var, [gensym() for _ in 1:(2*MAX + 4)])
#     output_symbols = [gensym() for _ in 1:2*(MAX - MIN + 1)]

#     # create calls
#     expand_call = :( ($(state_symbols[3]), $(state_symbols[4])) = model.input_layer($(state_symbols[1]), $(state_symbols[2])) )
#     state_calls = [:( ($(state_symbols[2*i+3]), $(state_symbols[2*i+4])) = model.hidden_layers[$i]($(state_symbols[2*i+1]), $(state_symbols[2*i+2])) ) for i in 1:MAX+1]
#     output_calls = [:( ($(output_symbols[2*i-1]), $(output_symbols[2*i])) = model.output_layers[$(MIN+i)]($(state_symbols[2*(MIN + i) + 3]), $(state_symbols[2*(MIN + i ) + 4])) ) for i in 1:(MAX - MIN + 1)]
#     return_call = :( return $(join_as_tuples(convert_to_tuples(output_symbols))) )
#     calls = vcat(expand_call, state_calls, output_calls, return_call)

#     return Expr(:block, calls...)

# end

convert_to_tuples(x) = ntuple(i -> :(($(x[2*i-1]), $(x[2*i]))), div(length(x),2))
join_as_tuples(x) = :( tuple($(x...) ))

end # module UnboundedBNN