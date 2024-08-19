export LinearSpike

using Zygote
using Random: default_rng

struct LinearSpike{T}
    W_mean :: Matrix{T}
    W_wstd :: Matrix{T}
    b_mean :: Vector{T}
    b_wstd :: Vector{T}
    zW :: Matrix{T}
    zb :: Vector{T}
end
@functor LinearSpike

function LinearSpike(p::Pair; initializer=(0,1), T=Float32, eps=1f0, rng=default_rng())
    return LinearSpike(
        initializer[1] .+ eps*randn(rng, T, p[2], p[1]), 
        convert(T, log(exp(initializer[2])-1)) .+ eps*randn(rng, T, p[2], p[1]),
        initializer[1] .+ eps*randn(rng, T, p[2]),
        convert(T, log(exp(initializer[2])-1)) .+ eps*randn(rng, T, p[2]),
        zeros(T, p[2], p[1]), 
        zeros(T, p[2])
    )
end

function (l::LinearSpike{T})(x; rng=default_rng()) where { T }

    zW = sigmoid.(l.zW)
    zb = sigmoid.(l.zb)

    W_mask = to_mask(zW)[:,:,1]
    b_mask = to_mask(zb)[:,1]

    W_std = softplus.(l.W_wstd)
    b_std = softplus.(l.b_wstd)
    W = (l.W_mean + W_std .* randn(rng, T, size(l.W_mean))) .* W_mask
    b = (l.b_mean + b_std .* randn(rng, T, size(l.b_mean))) .* b_mask

    return W * x .+ b
end

function KL_loss(l::LinearSpike)
    # ASSUMES STANDARD NORMAL PRIOR AND VAGUE BERNOULLI
    W_var = softplus.(l.W_wstd).^2
    b_var = softplus.(l.b_wstd).^2
    zW = sigmoid.(l.zW)
    zb = sigmoid.(l.zb)

    kl_w = sum(zW .* KL_normals.(l.W_mean, W_var))
    kl_b = sum(zb .* KL_normals.(l.b_mean, b_var)) 
    kl_zW = sum(zW .* log.(zW) + (1 .- zW) .* log.(1 .- zW))
    kl_zb = sum(zb .* log.(zb) + (1 .- zb) .* log.(1 .- zb))
    return kl_w + kl_b + kl_zW + kl_zb
end

function sample_gumbel(; epsilon=1e-10, T=Float32, rng=default_rng())
    # ret = rand(Float32, size...)
    # ret = -log.(-log.(ret .+ epsilon) .+ epsilon)
    ret1 = -log(-log(rand(rng, T) + epsilon) + epsilon)
    ret2 = -log(-log(rand(rng, T) + epsilon) + epsilon)
    return ret1, ret2
end

function sample_one_hot(p; epsilon=1e-10, tau=0.1)
    # logits = log.([p, 1-p] .+ epsilon)
    # y = logits + sample_gumbel(2, epsilon = epsilon)
    y = (log(p + epsilon), log(1 - p + epsilon)) .+ sample_gumbel(epsilon = epsilon)
    y_soft = UnboundedBNN.softmax([y ./ tau...])
    y_hard = (y_soft .== maximum(y_soft))
    ret = y_hard - Zygote.ChainRulesCore.ignore_derivatives(y_soft) + y_soft # return y_hard, but propagate gradients through y_soft
    return ret
end

function to_mask(p; epsilon=1e-10, tau=0.1, rng=default_rng())
    g1 = -log.(-log.(rand(rng, Float32, size(p)) .+ epsilon) .+ epsilon) 
    g2 = -log.(-log.(rand(rng, Float32, size(p)) .+ epsilon) .+ epsilon)

    y1 = (g1 .+ log.(p .+ epsilon)) ./ tau
    y2 = (g2 .+ log.(1 .- p .+ epsilon)) ./ tau

    y_soft = softmax(cat(y1, y2; dims=ndims(y1)+1), dims=ndims(y1)+1)
    y_hard = (y_soft .== maximum(y_soft, dims=ndims(y1)+1))
    ret = y_hard - Zygote.ChainRulesCore.ignore_derivatives(y_soft) + y_soft # return y_hard, but propagate gradients through y_soft
    return ret

end