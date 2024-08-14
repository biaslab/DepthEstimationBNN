export TruncatedPoisson, SafeTruncatedPoisson

import Statistics: mean, var

struct TruncatedPoisson{N,M,T<:Real}
    # assumes 95% quantile
    λ::T
    TruncatedPoisson(λ::T) where {T} = new{min_support(λ), max_support(λ),T}(λ)
end
@functor TruncatedPoisson (λ, )

struct SafeTruncatedPoisson{N,M,T}
    # assumes 95% quantile
    iλ::T
end
function SafeTruncatedPoisson(iλ::T; warn=true) where { T <: AbstractArray }
    return SafeTruncatedPoisson{min_support(softplus(iλ[1])), max_support(softplus(iλ[1])), T}(iλ)
end
function SafeTruncatedPoisson(iλ::T; warn=true) where { T <: Real }
    warn ? (@warn "SafeTruncatedPoisson initialized with scalar. Zygote will not be able to update scalars. Use arrays instead, as in SafeTruncatedPoisson([iλ]).") : nothing
    return SafeTruncatedPoisson{min_support(softplus(iλ)), max_support(softplus(iλ)), T}(iλ)
end
@functor SafeTruncatedPoisson (iλ, )

get_λ(p::TruncatedPoisson) = p.λ
get_iλ(p::TruncatedPoisson{N,M,<:Real}) where { N, M } = invsoftplus(p.λ)

get_iλ(p::SafeTruncatedPoisson{N,M,<:Real}) where { N, M } = p.iλ
get_iλ(p::SafeTruncatedPoisson{N,M,<:AbstractArray}) where { N, M } = p.iλ[1]
get_λ(p::SafeTruncatedPoisson) = softplus(get_iλ(p))

# min_support(λ) = Int(max(floor(λ - log(2)), 0)) # bound on median!
min_support(λ) = 0
max_support(λ) = Int(ceil(1.3 * λ + 5))
min_support(p::Union{TruncatedPoisson, SafeTruncatedPoisson}) = min_support(get_λ(p))
max_support(p::Union{TruncatedPoisson, SafeTruncatedPoisson}) = max_support(get_λ(p))
support(p) = min_support(p):max_support(p)

function normalization_constant(p::Union{TruncatedPoisson, SafeTruncatedPoisson})
    sum = 0
    dist = Poisson(get_λ(p))
    for x in support(p)
        sum += pdf(dist, x)
    end
    return sum
end
lognormalization_constant(p::Union{TruncatedPoisson, SafeTruncatedPoisson}) = log(normalization_constant(p))

pdf(p::Union{TruncatedPoisson, SafeTruncatedPoisson}, x::Int) = exp(logpdf(p, x))

function logpdf(p::Union{TruncatedPoisson, SafeTruncatedPoisson}, x::Int) 
    if x in support(p)
        return logpdf(Poisson(get_λ(p)), x) - lognormalization_constant(p)
    else
        return -Inf
    end
end

KL_loss(p::Union{TruncatedPoisson, SafeTruncatedPoisson}, q::Poisson) = sum(x -> pdf(p, x) * (logpdf(p, x) - logpdf(q, x)), support(p))