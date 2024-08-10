export Poisson

import Statistics: mean, var

struct Poisson{T}
    λ::T
end

mean(p::Poisson) = p.λ
var(p::Poisson) = p.λ

pdf(p::Poisson, x::Int) = x >= 0 ? exp(-p.λ) * p.λ^x / factorial(x) : 0

KL_loss(p::Poisson, q::Poisson) = p.λ * (log(p.λ) - log(q.λ)) + q.λ - p.λ