@testitem "Truncated distributions" begin

    using UnboundedBNN: TruncatedDistribution, pdf, cdf, Normal, normalization_constant, invcdf, get_dist, get_lower, get_upper, truncate, expand_truncation_to_ints

    @testset "get" begin
        @test get_dist(TruncatedDistribution(Normal(0, 1), 0, Inf)) == Normal(0, 1)
        @test get_dist(TruncatedDistribution(Normal(0, 1), -Inf, 0)) == Normal(0, 1)
        @test get_dist(TruncatedDistribution(Normal(2,3), 2, Inf)) == Normal(2, 3)
        @test get_lower(TruncatedDistribution(Normal(0, 1), 0, Inf)) == 0
        @test get_lower(TruncatedDistribution(Normal(0, 1), -Inf, 0)) == -Inf
        @test get_lower(TruncatedDistribution(Normal(2,3), 2, Inf)) == 2
        @test get_upper(TruncatedDistribution(Normal(0, 1), 0, Inf)) == Inf
        @test get_upper(TruncatedDistribution(Normal(0, 1), -Inf, 0)) == 0
        @test get_upper(TruncatedDistribution(Normal(2,3), 2, Inf)) == Inf
    end

    @testset "normalization_constant" begin
        @test normalization_constant(TruncatedDistribution(Normal(0, 1), -Inf, Inf)) ≈ 1
        @test normalization_constant(TruncatedDistribution(Normal(0, 1), -Inf, 0)) ≈ 0.5
        @test normalization_constant(TruncatedDistribution(Normal(2,3), 2, Inf)) ≈ 0.5
    end
        
    @testset "pdf" begin
        @test pdf(TruncatedDistribution(Normal(0, 1), 0, Inf), -1) ≈ 0
        @test pdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 0) ≈ 2 / sqrt(2π)
        @test pdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 1) ≈ 2 / sqrt(2π) * exp(-0.5)
        @test pdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 2) ≈ 2 / sqrt(2π) * exp(-2)

        @test pdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), -1) ≈ 2 / sqrt(2π) * exp(-0.5)
        @test pdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), 0) ≈ 2 / sqrt(2π)
        @test pdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), 1) ≈ 0

        @test pdf(TruncatedDistribution(Normal(2,3), 2, Inf), 1) ≈ 0
        @test pdf(TruncatedDistribution(Normal(2,3), 2, Inf), 2) ≈ 2 / sqrt(2π) / 3
        @test pdf(TruncatedDistribution(Normal(2,3), 2, Inf), 3) ≈ 2 / sqrt(2π) / 3 * exp(-0.5/9)
    end
    
    @testset "cdf" begin
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), -1) ≈ 0
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 0) ≈ 0
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 2) ≈ 1 - 2 * cdf(Normal(0, 1), -2)
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), Inf) ≈ 1
    end

    @testset "inv_cdf" begin
        @test invcdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 0) ≈ 0
        @test invcdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 1) ≈ Inf
        @test invcdf(TruncatedDistribution(Normal(3, 5), -1, 7), 0.0) ≈ -1
        @test invcdf(TruncatedDistribution(Normal(3, 5), -1, 7), 0.5) ≈ 3
        @test invcdf(TruncatedDistribution(Normal(3, 5), -1, 7), 1.0) ≈ 7
    end
        
    @testset "truncate" begin
        @test truncate(Normal(0, 1), 0, Inf) == TruncatedDistribution(Normal(0, 1), 0, Inf)
        @test truncate(TruncatedDistribution(Normal(0, 1), 0, Inf), 0, Inf) == TruncatedDistribution(Normal(0, 1), 0, Inf)
        @test truncate(Normal(0, 1), -Inf, 0) == TruncatedDistribution(Normal(0, 1), -Inf, 0)
        @test truncate(TruncatedDistribution(Normal(0, 1), -Inf, 0), -Inf, 0) == TruncatedDistribution(Normal(0, 1), -Inf, 0)
        @test truncate(TruncatedDistribution(Normal(2, 3), 2, Inf), 3, 8) == TruncatedDistribution(Normal(2, 3), 3, 8.0)
        @test truncate(TruncatedDistribution(Normal(2, 3), 3, 8), 2, Inf) == TruncatedDistribution(Normal(2, 3), 3, 8.0)
    end

    @testset "expand_truncation_to_ints" begin
        @test expand_truncation_to_ints(TruncatedDistribution(Normal(0, 1), 0, 5.4)) == TruncatedDistribution(Normal(0, 1), 0, 6)
        @test expand_truncation_to_ints(TruncatedDistribution(Normal(0, 1), -5.4, 0)) == TruncatedDistribution(Normal(0, 1), -6, 0)
        @test expand_truncation_to_ints(TruncatedDistribution(Normal(0, 1), -5.4, 5.4)) == TruncatedDistribution(Normal(0, 1), -6, 6)
    end

end