export Poisson, SafePoisson

using SpecialFunctions: logfactorial
import Statistics: mean, var, std

struct Poisson{T} <: Distribution
    λ::T
end
struct SafePoisson{T} <: Distribution
    transformed_λ::Vector{T}
end
@functor SafePoisson

UnionPoisson = Union{Poisson, SafePoisson}

get_λ(d::Poisson) = d.λ
get_λ(d::SafePoisson) = softplus(d.transformed_λ[1])

realtype(::Type{Poisson{T}}) where {T} = T
realtype(::Type{SafePoisson{T}}) where {T} = T
realtype(::Poisson{T}) where {T} = T
realtype(::SafePoisson{T}) where {T} = T

mean(p::UnionPoisson) = get_λ(p)
var(p::UnionPoisson) = get_λ(p)

function logpmf(p::UnionPoisson, x::Int) 
    if x < 0
        return -Inf
    else
        λ = get_λ(p)
        return convert(realtype(p), -λ + x * log(λ) - logfactorial(x))
    end
end
function pmf(p::UnionPoisson, x::Int)
    if x < 0
        return zero(realtype(p))
    else
        return exp(logpmf(p, x))
    end
end

cdf(p::UnionPoisson, x::Int) = mapreduce(k -> pmf(p, k), +, 0:x)
function invcdf(p::UnionPoisson, q::Real)
    for k in 0:1000
        if cdf(p, k) >= q
            return k
        end
    end
end

function KL_loss(p::UnionPoisson, q::UnionPoisson)
    pλ = get_λ(p)
    qλ = get_λ(q)
    return qλ - pλ + pλ * (log(pλ) - log(qλ))
end

mean(d::TruncatedDistribution{<:UnionPoisson}) = sum(x -> x * pmf(d, x), get_lower(d):get_upper(d))
function var(d::TruncatedDistribution{<:UnionPoisson})
    m = mean(d)
    return sum(x -> (x - m)^2 * pmf(d, x), get_lower(d):get_upper(d))
end
std(d::TruncatedDistribution{<:UnionPoisson}) = sqrt(var(d))

function KL_loss(p::TruncatedDistribution{<:UnionPoisson}, q::UnionPoisson) 
    if get_dist(p) == q
        return zero(promote_type(realtype(p), realtype(q)))
    end
    lower, upper = support(p)
    return sum(x -> pmf(p, x) * (log(pmf(p, x) / pmf(q, x))), lower:upper)
end