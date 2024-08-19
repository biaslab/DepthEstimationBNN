export Linear

using Random: default_rng

struct Linear{T}
    W_mean :: Matrix{T}
    W_wvar :: Matrix{T}
    b_mean :: Vector{T}
    b_wvar :: Vector{T}
end
@functor Linear

function Linear(p::Pair; initializer=(0,1), T=Float32, eps=1f0, rng=default_rng())
    return Linear(
        initializer[1] .+ eps*randn(rng, T, p[2], p[1]), 
        convert(T, log(exp(initializer[2])-1)) .+ eps*randn(rng, T, p[2], p[1]),
        initializer[1] .+ eps*randn(rng, T, p[2]),
        convert(T, log(exp(initializer[2])-1)) .+ eps*randn(rng, T, p[2])
    )
end

function (l::Linear)(x_mean, x_var)
    W_var = softplus.(l.W_wvar)
    b_var = softplus.(l.b_wvar)

    y_mean = l.W_mean * x_mean + l.b_mean
    y_var = W_var * x_var + l.W_mean.^2 * x_var + W_var * x_mean.^2 + b_var
    return y_mean, y_var
end

KL_normals(m, v) = ( -log(v) + v + abs2(m) - 1 ) / 2

function KL_loss(l::Linear)
    # ASSUMES STANDARD NORMAL PRIOR
    W_var = softplus.(l.W_wvar)
    b_var = softplus.(l.b_wvar)
    return sum(KL_normals.(l.W_mean, W_var)) + sum(KL_normals.(l.b_mean, b_var)) 
end