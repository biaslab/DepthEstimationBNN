@testitem "Poisson distribution" begin

    using UnboundedBNN: Poisson, SafePoisson, mean, var, std, cdf, pdf, pmf, logpmf, KL_loss, get_λ, realtype, TruncatedDistribution

    construct_safepoisson(λ) = SafePoisson([invsoftplus(λ)])

    for poisson_constructor in (Poisson, construct_safepoisson)

        @test get_λ(poisson_constructor(1.0)) ≈ 1.0

        @test realtype(poisson_constructor(1.0)) == Float64
        @test realtype(typeof(poisson_constructor(1.0))) == Float64
        @test realtype(poisson_constructor(1.f0)) == Float32
        @test realtype(typeof(poisson_constructor(1.f0))) == Float32

        @test mean(poisson_constructor(1.0)) ≈ 1.0
        @test mean(poisson_constructor(2.0)) ≈ 2.0
        @test std(poisson_constructor(1.0)) ≈ 1.0
        @test std(poisson_constructor(2.0)) ≈ sqrt(2.0)
        @test var(poisson_constructor(1.0)) ≈ 1.0
        @test var(poisson_constructor(2.0)) ≈ 2.0

        @test logpmf(poisson_constructor(1.0), 0) ≈ -1.0
        @test logpmf(poisson_constructor(1.0), 1) ≈ -1.0
        @test logpmf(poisson_constructor(1.0), 2) ≈ -1.0 - log(2.0)
        @test logpmf(poisson_constructor(1.0), 3) ≈ -1.0 - log(6.0)
        @test logpmf(poisson_constructor(1.0), -1) ≈ -Inf
        @test logpmf(poisson_constructor(1.0), -2) ≈ -Inf

        @test logpmf(poisson_constructor(2.0), 0) ≈ -2.0
        @test logpmf(poisson_constructor(2.0), 1) ≈ -2.0 + log(2.0)
        @test logpmf(poisson_constructor(2.0), 2) ≈ -2.0 + log(4.0) - log(2.0)
        @test logpmf(poisson_constructor(2.0), 3) ≈ -2.0 + log(8.0) - log(6.0)
        @test logpmf(poisson_constructor(2.0), -1) ≈ -Inf
        @test logpmf(poisson_constructor(2.0), -2) ≈ -Inf

        @test pmf(poisson_constructor(1.0), 0) ≈ exp(-1.0)
        @test pmf(poisson_constructor(1.0), 1) ≈ exp(-1.0)
        @test pmf(poisson_constructor(1.0), 2) ≈ exp(-1.0) / 2.0
        @test pmf(poisson_constructor(1.0), 3) ≈ exp(-1.0) / 6.0
        @test pmf(poisson_constructor(1.0), -1) ≈ 0
        @test pmf(poisson_constructor(1.0), -2) ≈ 0

        @test pmf(poisson_constructor(2.0), 0) ≈ exp(-2.0)
        @test pmf(poisson_constructor(2.0), 1) ≈ exp(-2.0) * 2.0
        @test pmf(poisson_constructor(2.0), 2) ≈ exp(-2.0) * 4.0 / 2.0
        @test pmf(poisson_constructor(2.0), 3) ≈ exp(-2.0) * 8.0 / 6.0
        @test pmf(poisson_constructor(2.0), -1) ≈ 0
        @test pmf(poisson_constructor(2.0), -2) ≈ 0
        @test pmf(poisson_constructor(3.0), 500) ≈ 0.0

        @test pdf(poisson_constructor(1.0), 0.5) ≈ 0.0

        @test cdf(poisson_constructor(1.0), 0) ≈ exp(-1.0)
        @test cdf(poisson_constructor(1.0), 1) ≈ 2 * exp(-1.0)
        @test cdf(poisson_constructor(1.0), 100) ≈ 1.0

        @test KL_loss(poisson_constructor(1.0), poisson_constructor(1.0)) ≈ 0.0
        @test KL_loss(poisson_constructor(1.0), poisson_constructor(2.0)) ≈ - log(2) + 1
        @test KL_loss(poisson_constructor(2.0), poisson_constructor(1.0)) ≈ 2 * log(2) - 1
        @test KL_loss(poisson_constructor(2.0), poisson_constructor(2.0)) ≈ 0.0

        @test mean(TruncatedDistribution(poisson_constructor(1.0), 0, 0)) ≈ 0.0
        @test mean(TruncatedDistribution(poisson_constructor(1.0), 0, 1)) ≈ 0.5
        @test mean(TruncatedDistribution(poisson_constructor(1.0), 0, 2)) ≈ 0.8

        @test var(TruncatedDistribution(poisson_constructor(1.0), 0, 0)) ≈ 0.0
        @test var(TruncatedDistribution(poisson_constructor(1.0), 0, 1)) ≈ 0.25
        @test var(TruncatedDistribution(poisson_constructor(1.0), 0, 2)) ≈ 14/25

        @test std(TruncatedDistribution(poisson_constructor(1.0), 0, 0)) ≈ 0.0
        @test std(TruncatedDistribution(poisson_constructor(1.0), 0, 1)) ≈ 0.5
        @test std(TruncatedDistribution(poisson_constructor(1.0), 0, 2)) ≈ sqrt(14/25)

    end

end