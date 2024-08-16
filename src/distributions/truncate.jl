export TruncatedDistribution
export truncate, truncate_to_quantiles, expand_truncation_to_ints

struct TruncatedDistribution{D <: Distribution, T1 <: Real, T2 <: Real} <: Distribution
    dist::D
    lower::T1
    upper::T2
end
@functor TruncatedDistribution (dist, )

get_dist(d::TruncatedDistribution) = d.dist
get_lower(d::TruncatedDistribution) = d.lower
get_upper(d::TruncatedDistribution) = d.upper
support(d::TruncatedDistribution) = get_lower(d), get_upper(d)

normalization_constant(d::TruncatedDistribution) = cdf(get_dist(d), get_upper(d)) - cdf(get_dist(d), get_lower(d))

function pdf(d::TruncatedDistribution, x::T) where { T <: Real }
    if x < get_lower(d) || x > get_upper(d)
        return zero(T)
    end
    return pdf(get_dist(d), x) / normalization_constant(d)
end
function cdf(d::TruncatedDistribution, x::T) where { T <: Real }
    if x <= get_lower(d)
        return zero(T)
    elseif x >= get_upper(d)
        return one(T)
    end
    return (cdf(get_dist(d), x) - cdf(get_dist(d), get_lower(d))) / normalization_constant(d)
end
function invcdf(d::TruncatedDistribution, p)
    return invcdf(get_dist(d), cdf(get_dist(d), get_lower(d)) + p * normalization_constant(d))
end

truncate(d::Distribution, lower::Real, upper::Real) = TruncatedDistribution(d, lower, upper)
truncate(d::TruncatedDistribution, lower::Real, upper::Real) = TruncatedDistribution(get_dist(d), max(get_lower(d), lower), min(get_upper(d), upper))

function truncate_to_quantiles(d::Distribution, lower_quantile::Real, upper_quantile::Real)
    lower = invcdf(d, lower_quantile)
    upper = invcdf(d, upper_quantile)
    return truncate(d, lower, upper)
end

function expand_truncation_to_ints(d::TruncatedDistribution)
    return TruncatedDistribution(get_dist(d), floor(Int, get_lower(d)), ceil(Int, get_upper(d)))
end