export Normal, SafeNormal

using StatsFuns: normpdf

struct Normal{T} <: Distribution
    μ::T
    σ::T
end
struct SafeNormal{T} <: Distribution
    μ::Vector{T}
    transformed_σ::Vector{T}
end
@functor SafeNormal

UnionNormal = Union{Normal, SafeNormal}

get_μ(d::Normal) = d.μ
get_μ(d::SafeNormal) = d.μ[1]
get_σ(d::Normal) = d.σ
get_σ(d::SafeNormal) = softplus(d.transformed_σ[1])

support(d::Normal) = -Inf, Inf
support(d::SafeNormal) = -Inf, Inf

pdf(d::UnionNormal, x::Real) = normpdf(get_μ(d), get_σ(d), x)
logpdf(d::UnionNormal, x::Real) = normlogpdf(get_μ(d), get_σ(d), x)
cdf(d::UnionNormal, x::Real) = normcdf(get_μ(d), get_σ(d), x)
invcdf(d::UnionNormal, p::Real) = norminvcdf(get_μ(d), get_σ(d), p)