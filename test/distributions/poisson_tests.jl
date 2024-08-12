@testitem "Poisson distribution" begin

    using UnboundedBNN: Poisson, mean, var, pdf, logpdf, KL_loss

    @test mean(Poisson(1.0)) ≈ 1.0
    @test mean(Poisson(2.0)) ≈ 2.0
    @test var(Poisson(1.0)) ≈ 1.0
    @test var(Poisson(2.0)) ≈ 2.0

    @test pdf(Poisson(1.0), 0) ≈ exp(-1.0)
    @test pdf(Poisson(1.0), 1) ≈ exp(-1.0)
    @test pdf(Poisson(1.0), 2) ≈ exp(-1.0) / 2.0
    @test pdf(Poisson(1.0), 3) ≈ exp(-1.0) / 6.0
    @test pdf(Poisson(1.0), -1) ≈ 0
    @test pdf(Poisson(1.0), -2) ≈ 0

    @test pdf(Poisson(2.0), 0) ≈ exp(-2.0)
    @test pdf(Poisson(2.0), 1) ≈ exp(-2.0) * 2.0
    @test pdf(Poisson(2.0), 2) ≈ exp(-2.0) * 4.0 / 2.0
    @test pdf(Poisson(2.0), 3) ≈ exp(-2.0) * 8.0 / 6.0
    @test pdf(Poisson(2.0), -1) ≈ 0
    @test pdf(Poisson(2.0), -2) ≈ 0
    @test pdf(Poisson(3.0), 500) ≈ 0.0

    @test KL_loss(Poisson(1.0), Poisson(1.0)) ≈ 0.0
    @test KL_loss(Poisson(1.0), Poisson(2.0)) ≈ - log(2) + 1
    @test KL_loss(Poisson(2.0), Poisson(1.0)) ≈ 2 * log(2) - 1
    @test KL_loss(Poisson(2.0), Poisson(2.0)) ≈ 0.0

end