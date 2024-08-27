@testitem "Normal distribution" begin

    using UnboundedBNN: Normal, SafeNormal, get_μ, get_σ, realtype, support, pdf, logpdf, logcdf, logccdf, ccdf, cdf, invcdf, invsoftplus
    
    safe_normal_constructor(μ, σ) = SafeNormal([μ], [invsoftplus(σ)])

    for normal_constructor in (Normal, safe_normal_constructor)
        
        d1 = normal_constructor(0.0, 1.0)
        d1 = Normal(0.0, 1.0)
        @test get_μ(d1) == 0.0
        @test get_σ(d1) == 1.0
        @test support(d1) == (-Inf, Inf)
        @test pdf(d1, 0.0) > 0
        @test logpdf(d1, 0.0) ≈ log(pdf(d1, 0.0))
        @test cdf(d1, -Inf) ≈ 0.0
        @test cdf(d1, 0.0) ≈ 0.5
        @test cdf(d1, Inf) ≈ 1.0
        @test logcdf(d1, Inf) ≈ 0.0
        @test logcdf(d1, -Inf) ≈ -Inf
        @test logccdf(d1, Inf) ≈ -Inf
        @test logccdf(d1, -Inf) ≈ 0.0
        @test logcdf(d1, 0.0) ≈ log(0.5)
        @test logccdf(d1, 0.0) ≈ log(0.5)
        @test invcdf(d1, 0.0) ≈ -Inf
        @test invcdf(d1, 0.5) ≈ 0.0
        @test invcdf(d1, 1.0) ≈ Inf
        @test realtype(typeof(d1)) == Float64
        @test realtype(d1) == Float64
        @test KL_loss(d1, d1) ≈ 0.0

        d2 = normal_constructor(1.0, 2.0)
        @test get_μ(d2) == 1.0
        @test get_σ(d2) == 2.0
        @test support(d2) == (-Inf, Inf)
        @test pdf(d2, 1.0) > 0
        @test logpdf(d2, 1.0) ≈ log(pdf(d2, 1.0))
        @test cdf(d2, -Inf) ≈ 0.0
        @test cdf(d2, 1.0) ≈ 0.5
        @test cdf(d2, Inf) ≈ 1.0
        @test invcdf(d2, 0.0) ≈ -Inf
        @test invcdf(d2, 0.5) ≈ 1.0
        @test invcdf(d2, 1.0) ≈ Inf
        @test KL_loss(d2, d2) ≈ 0.0

    end

end