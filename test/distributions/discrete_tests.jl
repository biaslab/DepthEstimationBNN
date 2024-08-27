@testitem "DiscreteDistribution" begin

    using UnboundedBNN: DiscreteDistribution, get_dist, support, pmf, logpmf, H, truncate, Normal, cdf, mean, var, std, realtype

    d = DiscreteDistribution(Normal(1.0, 2.0))
    @test get_dist(d) == Normal(1.0, 2.0)
    @test discretize(Normal(1.0, 2.0)) == d
    for k in -10:10
        @test pmf(d, k) >= 0
    end
    @test realtype(DiscreteDistribution(Normal(1.0, 2.0))) == Float64
    @test realtype(typeof(DiscreteDistribution(Normal(1.0, 2.0)))) == Float64
    @test realtype(DiscreteDistribution(Normal(1.0f0, 2.0f0))) == Float32
    @test realtype(typeof(DiscreteDistribution(Normal(1.0f0, 2.0f0)))) == Float32
    
    @testset "support" begin
        @test support(DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 5.0))) == (0, 4)
        @test support(DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 1.0))) == (0, 0)
        @test support(DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 1.1))) == (0, 1)
        @test support(DiscreteDistribution(truncate(Normal(0.0, 1.0), -0.1, 1.5))) == (-1, 1)
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
        @test pmf(d3, 0) > 0.0
        @test pmf(d3, 4) > 0.0
        @test pmf(d3, 5) ≈ 0.0
        @test pmf(d3, 6) ≈ 0.0

        d4 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 1.0))
        @test pmf(d4, -1) ≈ 0.0
        @test pmf(d4, 0) ≈ 1.0
        @test pmf(d4, 1) ≈ 0.0

        d5 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 0.1))
        @test pmf(d5, -1) ≈ 0.0
        @test pmf(d5, 0) ≈ 1.0
        @test pmf(d5, 1) ≈ 0.0

        d6 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 1.5))
        @test pmf(d6, -1) ≈ 0.0
        @test pmf(d6, 0) > 0.0
        @test pmf(d6, 1) > 0.0
        @test pmf(d6, 2) ≈ 0.0
    end

    @testset "logpmf" begin
        d1 = DiscreteDistribution(Normal(1.0, 2.0))
        @test logpmf(d1, 0) ≈ log(cdf(Normal(1.0, 2.0), 1) - cdf(Normal(1.0, 2.0), 0))
        @test logpmf(d1, 1) ≈ log(cdf(Normal(1.0, 2.0), 2) - cdf(Normal(1.0, 2.0), 1))
        @test logpmf(d1, 2) ≈ log(cdf(Normal(1.0, 2.0), 3) - cdf(Normal(1.0, 2.0), 2))

        d2 = DiscreteDistribution(Normal(0.0, 1.0) * H())
        @test logpmf(d2, -1) ≈ -Inf
        @test logpmf(d2, 0) ≈ log(2 * (cdf(Normal(0.0, 1.0), 0) - cdf(Normal(0.0, 1.0), -1)))

        d3 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 5.0))
        @test logpmf(d3, -1) ≈ -Inf
        @test logpmf(d3, 0) > -Inf
        @test logpmf(d3, 4) > -Inf
        @test logpmf(d3, 5) ≈ -Inf
        @test logpmf(d3, 6) ≈ -Inf

        d4 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 1.0))
        @test logpmf(d4, -1) ≈ -Inf
        @test logpmf(d4, 0) ≈ 0.0
        @test logpmf(d4, 1) ≈ -Inf

        d5 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 0.1))
        @test logpmf(d5, -1) ≈ -Inf
        @test logpmf(d5, 0) ≈ 0.0
        @test logpmf(d5, 1) ≈ -Inf

        d6 = DiscreteDistribution(truncate(Normal(0.0, 1.0), 0.0, 1.5))
        @test logpmf(d6, -1) ≈ -Inf
        @test logpmf(d6, 0) > -Inf
        @test logpmf(d6, 1) > -Inf
    end

    @testset "robustness" begin
        d = DiscreteDistribution(Normal(0.0, 1.0))
        @show logpmf(d, -100)
        @test logpmf(d, -100) > -Inf
        @test logpmf(d, 100) > -Inf
        @test logpmf(d, -100) ≈ logpmf(d, 99)
    end

    @testset "statistics" begin
        dist = discretize(truncate(Normal(0, 2), 0, 1))
        @test mean(dist) ≈ 0.0
        @test var(dist) ≈ 0.0
        @test std(dist) ≈ 0.0

        dist = discretize(truncate(Normal(0, 2), 0, 2))
        @test mean(dist) ≈ pmf(dist, 1)
        @test var(dist) ≈ pmf(dist,1)  * ( (1-pmf(dist,1))^2 + pmf(dist,0) * pmf(dist,1))
        @test std(dist) ≈ sqrt(var(dist))
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