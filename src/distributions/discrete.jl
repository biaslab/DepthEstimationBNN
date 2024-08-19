export DiscreteDistribution
export discretize, pmf

struct DiscreteDistribution{D <: Distribution} <: Distribution
    dist::D
end
@functor DiscreteDistribution

get_dist(d::DiscreteDistribution) = d.dist
function support(d::DiscreteDistribution)
    lower, upper = support(get_dist(d))
    return floor(Int, lower), ceil(Int, upper - 1)
end

function pmf(d::DiscreteDistribution, x::Int)
    lower, upper = support(get_dist(d))
    if x < lower || x > upper
        return 0.0
    elseif upper-1 < x <= upper
        return (cdf(get_dist(d), upper) - cdf(get_dist(d), x))
    end
    return (cdf(get_dist(d), x+1) - cdf(get_dist(d), x))
end

discretize(d::Distribution) = DiscreteDistribution(d)

function KL_loss(p::DiscreteDistribution, q::DiscreteDistribution) 
    if get_dist(p) == get_dist(q)
        return 0.0
    end
    lower, upper = support(p)
    return sum(x -> pmf(p, x) * (log(pmf(p, x) / pmf(q, x))), lower:upper)
end