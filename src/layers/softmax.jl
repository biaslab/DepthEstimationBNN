export Softmax

using StatsFuns: softmax

struct Softmax{N}
    Softmax(N::Int) =  new{N}()
end

function (l::Softmax)(x_mean, x_var; rng=default_rng())
    tmp = randn(rng, size(x_var)) .* sqrt.(x_var) .+ x_mean
    return tmp .- logsumexp(tmp, dims=1), nothing
end

(l::Softmax)(x::Vector) = x .- logsumexp(x)
(l::Softmax)(x::Matrix) = x .- logsumexp(x, dims = 1)