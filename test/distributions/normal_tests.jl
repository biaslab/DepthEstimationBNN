@testitem "Normal distribution" begin

    using UnboundedBNN: Normal, SafeNormal, get_μ, get_σ, support, pdf, logpdf, cdf, invcdf, invsoftplus
    
    @testset "Standard Normal" begin
        d = Normal(0.0, 1.0)
        @test get_μ(d) == 0.0
        @test get_σ(d) == 1.0
        @test support(d) == (-Inf, Inf)
        @test pdf(d, 0.0) > 0
        @test logpdf(d, 0.0) ≈ log(pdf(d, 0.0))
        @test cdf(d, -Inf) ≈ 0.0
        @test cdf(d, 0.0) ≈ 0.5
        @test cdf(d, Inf) ≈ 1.0
        @test invcdf(d, 0.0) ≈ -Inf
        @test invcdf(d, 0.5) ≈ 0.0
        @test invcdf(d, 1.0) ≈ Inf
    end

    @testset "Normal" begin
        d = Normal(1.0, 2.0)
        @test get_μ(d) == 1.0
        @test get_σ(d) == 2.0
        @test support(d) == (-Inf, Inf)
        @test pdf(d, 1.0) > 0
        @test logpdf(d, 1.0) ≈ log(pdf(d, 1.0))
        @test cdf(d, -Inf) ≈ 0.0
        @test cdf(d, 1.0) ≈ 0.5
        @test cdf(d, Inf) ≈ 1.0
        @test invcdf(d, 0.0) ≈ -Inf
        @test invcdf(d, 0.5) ≈ 1.0
        @test invcdf(d, 1.0) ≈ Inf
    end
    
    @testset "Standard SafeNormal" begin
        d = SafeNormal([0.0], [invsoftplus(1.0)])
        @test get_μ(d) == 0.0
        @test get_σ(d) == 1.0
        @test support(d) == (-Inf, Inf)
        @test pdf(d, 0.0) > 0
        @test logpdf(d, 0.0) ≈ log(pdf(d, 0.0))
        @test cdf(d, -Inf) ≈ 0.0
        @test cdf(d, 0.0) ≈ 0.5
        @test cdf(d, Inf) ≈ 1.0
        @test invcdf(d, 0.0) ≈ -Inf
        @test invcdf(d, 0.5) ≈ 0.0
        @test invcdf(d, 1.0) ≈ Inf
    end
    
    @testset "SafeNormal" begin
        d = SafeNormal([1.0], [invsoftplus(2.0)])
        @test get_μ(d) == 1.0
        @test get_σ(d) == 2.0
        @test support(d) == (-Inf, Inf)
        @test pdf(d, 1.0) > 0
        @test logpdf(d, 1.0) ≈ log(pdf(d, 1.0))
        @test cdf(d, -Inf) ≈ 0.0
        @test cdf(d, 1.0) ≈ 0.5
        @test cdf(d, Inf) ≈ 1.0
        @test invcdf(d, 0.0) ≈ -Inf
        @test invcdf(d, 0.5) ≈ 1.0
        @test invcdf(d, 1.0) ≈ Inf
    end

end