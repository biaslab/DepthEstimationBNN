export TruncatedPoisson

import Statistics: mean, var

struct TruncatedPoisson{N,M,T}
    # assumes 95% quantile
    iλ::T
    # TruncatedPoisson(λ::T) where {T} = new{min_support(softplus(λ[1])), max_support(softplus(λ[1])),T}(λ)
end
@functor TruncatedPoisson (iλ, )

function TruncatedPoisson(λ::T) where { T <: AbstractArray }
    return TruncatedPoisson{min_support(λ[1]), max_support(λ[1]), T}(invsoftplus.(λ))
end
function TruncatedPoisson(λ::T; warn=true) where { T <: Real }
    warn ? (@warn "TruncatedPoisson initialized with scalar. Zygote will not be able to update scalars. Use arrays instead, as in TruncatedPoisson([λ]).") : nothing
    return TruncatedPoisson{min_support(λ), max_support(λ), T}(invsoftplus(λ))
end

get_iλ(p::TruncatedPoisson{N,M,<:Real}) where { N, M } = p.iλ
get_iλ(p::TruncatedPoisson{N,M,<:AbstractArray}) where { N, M } = p.iλ[1]
get_λ(p::TruncatedPoisson) = softplus(get_iλ(p))

min_support(λ) = Int(max(floor(λ - log(2)), 0))
max_support(λ) = Int(ceil(1.3 * λ + 5))
min_support(p::TruncatedPoisson) = min_support(get_λ(p))
max_support(p::TruncatedPoisson) = max_support(get_λ(p))
support(p::TruncatedPoisson) = min_support(p):max_support(p)

function normalization_constant(p::TruncatedPoisson)
    sum = 0
    dist = Poisson(get_λ(p))
    for x in support(p)
        sum += pdf(dist, x)
    end
    return sum
end

function pdf(p::TruncatedPoisson, x::Int) 
    if x in support(p)
        return pdf(Poisson(get_λ(p)), x) / normalization_constant(p)
    else
        return 0
    end
end

KL_loss(p::TruncatedPoisson, ::Poisson) = - log(normalization_constant(p))