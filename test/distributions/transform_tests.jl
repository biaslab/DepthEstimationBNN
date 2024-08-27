@testitem "Transformed Distribution" begin
    
    using UnboundedBNN: TransformedDistribution, transform, truncate, discretize, TruncatedDistribution, DiscreteDistribution

    d1 = TransformedDistribution(Normal(0, 1), x -> truncate(x, 0, 1))
    @test transform(d1) == TruncatedDistribution(Normal(0, 1), 0, 1)

    d2 = TransformedDistribution(
        Normal(0, 1),
        (
            x -> truncate(x, 0, 1),
            x -> discretize(x)
        )
    )
    @test transform(d2) == DiscreteDistribution(TruncatedDistribution(Normal(0, 1), 0, 1))
    @test transform(Normal(0, 1)) == Normal(0, 1)
end