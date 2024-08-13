export Softmax

using StatsFuns: softmax

struct Softmax{N} 
    Softmax(N::Int) =  new{N}()
end

function (l::Softmax)(x_mean, x_var)
    tmp = randn(2) .* sqrt.(x_var) .+ x_mean
    return tmp .- logsumexp(tmp), nothing
end

(l::Softmax)(x::Vector) = x .- logsumexp(x)
(l::Softmax)(x::Matrix) = x .- logsumexp(x, dims = 1)

