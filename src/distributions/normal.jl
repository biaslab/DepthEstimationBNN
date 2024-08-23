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

realtype(::Type{Normal{T}}) where {T} = T
realtype(::Type{SafeNormal{T}}) where {T} = T
realtype(::Normal{T}) where {T} = T
realtype(::SafeNormal{T}) where {T} = T

support(::Normal) = -Inf, Inf
support(::SafeNormal) = -Inf, Inf

pdf(d::UnionNormal, x::Real) = normpdf(get_μ(d), get_σ(d), x)
logpdf(d::UnionNormal, x::Real) = normlogpdf(get_μ(d), get_σ(d), x)
cdf(d::UnionNormal, x::Real) = normcdf(get_μ(d), get_σ(d), x)
invcdf(d::UnionNormal, p::Real) = norminvcdf(get_μ(d), get_σ(d), p)

KL_loss(d1::UnionNormal, d2::UnionNormal) = KL_normals(get_μ(d1), get_σ(d1)^2, get_μ(d2), get_σ(d2)^2)
KL_normals(m, v) = KL_normals(m, v, 0, 1)
KL_normals(pm, pv, qm, qv) = (log(qv/pv) + (pv + abs2(pm - qm) )/qv - 1)/2