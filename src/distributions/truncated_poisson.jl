export TruncatedPoisson

import Statistics: mean, var

struct TruncatedPoisson{N,M,T}
    # assumes 95% quantile
    λ::T
    TruncatedPoisson(λ::T) where {T} = new{min_support(softplus(λ[1])), max_support(softplus(λ[1])),T}(λ)
end
@functor TruncatedPoisson (λ, )

min_support(λ) = Int(max(floor(λ - log(2)), 0))
max_support(λ) = Int(ceil(1.3 * λ + 5))
min_support(p::TruncatedPoisson) = min_support(softplus(p.λ[1]))
max_support(p::TruncatedPoisson) = max_support(softplus(p.λ[1]))
support(p::TruncatedPoisson) = min_support(p):max_support(p)

function normalization_constant(p::TruncatedPoisson)
    sum = 0
    dist = Poisson(softplus(p.λ[1]))
    for x in min_support(p):max_support(p)
        sum += pdf(dist, x)
    end
    return sum
end

function pdf(p::TruncatedPoisson, x::Int) 
    if min_support(p) <= x <= max_support(p)
        return pdf(Poisson(softplus(p.λ[1])), x) / normalization_constant(p)
    else
        return 0
    end
end

function KL_loss(p::TruncatedPoisson, q::Poisson)
    support = (min_support(p)-1):(max_support(p)-1)
    pdf_q = pdf.(Ref(q), support)
    pdf_p = pdf_q ./ sum(pdf_q)
    return sum(pdf_p .* log.(pdf_p ./ pdf_q))
end