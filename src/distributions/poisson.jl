export Poisson

using SpecialFunctions: logfactorial
import Statistics: mean, var

struct Poisson{T}
    λ::T
end

mean(p::Poisson) = p.λ
var(p::Poisson) = p.λ

function logpdf(p::Poisson, x::Int)
    if x < 0
        return -Inf
    else
        # return -p.λ + x * log(p.λ) - log(factorial(x))
        return -p.λ + x * log(p.λ) - logfactorial(x)
    end
end
pdf(p::Poisson, x::Int) = exp(logpdf(p, x))

KL_loss(p::Poisson, q::Poisson) = p.λ * (log(p.λ) - log(q.λ)) + q.λ - p.λ