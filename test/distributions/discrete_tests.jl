@testitem "DiscreteDistribution" begin

    using UnboundedBNN: DiscreteDistribution, get_dist, support, pmf, H, truncate, Normal, cdf

    d = DiscreteDistribution(Normal(1.0, 2.0))
    @test get_dist(d) == Normal(1.0, 2.0)
    @test discretize(Normal(1.0, 2.0)) == d
    for k in -10:10
        @test pmf(d, k) >= 0
    end
    
    @testset "pmf" begin
        d1 = DiscreteDistribution(Normal(1.0, 2.0))
        @test pmf(d1, 0) ≈ (cdf(Normal(1.0, 2.0), 1) - cdf(Normal(1.0, 2.0), 0))
        @test pmf(d1, 1) ≈ (cdf(Normal(1.0, 2.0), 2) - cdf(Normal(1.0, 2.0), 1))
        @test pmf(d1, 2) ≈ (cdf(Normal(1.0, 2.0), 3) - cdf(Normal(1.0, 2.0), 2))

        d2 = DiscreteDistribution(Normal(0.0, 1.0) * H())
        @test pmf(d2, -1) ≈ 0.0
        @test pmf(d2, 0) ≈ 2 * (cdf(Normal(0.0, 1.0), 0) - cdf(Normal(0.0, 1.0), -1))

        d3 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 5.0))
        @test pmf(d3, -1) ≈ 0.0
        @test pmf(d3, 0) >= 0.0
        @test pmf(d3, 5) >= 0.0
        @test pmf(d3, 6) ≈ 0.0
    end

    @testset "KL_loss" begin
        d1 = DiscreteDistribution(Normal(1.0, 2.0))
        @test KL_loss(d1, d1) ≈ 0.0

        d2 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 5.0))
        @test KL_loss(d2, d2) ≈ 0.0
        @test KL_loss(d2, d1) >= 0.0

        d3 = discretize(truncate(Normal(0, 1), -5, 0))
        @test KL_loss(d2, d3) ≈ Inf
    end
    
end