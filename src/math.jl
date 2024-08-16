export H, Heaviside
export softplus, invsoftplus, sigmoid

const softplus = log1pexp
const invsoftplus = logexpm1
const sigmoid = logistic

struct ShiftedHeaviside{T}
    a::T
end

Heaviside() = return ShiftedHeaviside(0)

const H = Heaviside