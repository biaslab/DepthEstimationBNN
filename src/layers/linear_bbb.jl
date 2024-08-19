export LinearBBB

using Random: default_rng

struct LinearBBB{T}
    W_mean :: Matrix{T}
    W_wstd :: Matrix{T}
    b_mean :: Vector{T}
    b_wstd :: Vector{T}
end
@functor LinearBBB

function LinearBBB(p::Pair; initializer=(0,1), T=Float32, eps=1f0, rng=default_rng())
    return LinearBBB(
        initializer[1] .+ eps*randn(rng, T, p[2], p[1]), 
        convert(T, log(exp(initializer[2])-1)) .+ eps*randn(rng, T, p[2], p[1]),
        initializer[1] .+ eps*randn(rng, T, p[2]),
        convert(T, log(exp(initializer[2])-1)) .+ eps*randn(rng, T, p[2])
    )
end

function (l::LinearBBB{T})(x; rng=default_rng()) where { T }
    W_std = softplus.(l.W_wstd)
    b_std = softplus.(l.b_wstd)

    W = l.W_mean + W_std .* randn(rng, T, size(l.W_mean))
    b = l.b_mean + b_std .* randn(rng, T, size(l.b_mean))
    y = W * x .+ b
    return y
end

function KL_loss(l::LinearBBB)
    # ASSUMES STANDARD NORMAL PRIOR
    W_var = softplus.(l.W_wstd).^2
    b_var = softplus.(l.b_wstd).^2
    return sum(KL_normals.(l.W_mean, W_var)) + sum(KL_normals.(l.b_mean, b_var)) 
end