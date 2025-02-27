@testitem "Truncated distributions" begin

    using UnboundedBNN: TruncatedDistribution, pdf, logcdf, cdf, Normal, lognormalization_constant, normalization_constant, invcdf, get_dist, get_lower, get_upper, truncate, expand_truncation_to_ints, realtype, support

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

    @testset "support" begin
        @test support(TruncatedDistribution(Normal(0, 1), 0, Inf)) == (0, Inf)
        @test support(TruncatedDistribution(Normal(0, 1), -Inf, 0)) == (-Inf, 0)
        @test support(TruncatedDistribution(Normal(2,3), 2, Inf)) == (2, Inf)
        @test support(TruncatedDistribution(Normal(2,3), 2, 3)) == (2, 3)
    end

    @testset "realtype" begin
        @test realtype(TruncatedDistribution(Normal(0, 1), 0, Inf)) == Float64
        @test realtype(TruncatedDistribution(Normal(0, 1), -Inf, 0)) == Float64
        @test realtype(TruncatedDistribution(Normal(2,3), 2, Inf)) == Float64
        @test realtype(TruncatedDistribution(Normal(2,3), 2, 3)) == Int64
        @test realtype(TruncatedDistribution(Normal(2,3), 2, 3.0)) == Float64
        @test realtype(TruncatedDistribution(Normal(2f0,3f0), 2, 3.0)) == Float64
        @test realtype(TruncatedDistribution(Normal(2f0,3f0), 2, 3)) == Float32

        @test realtype(typeof(TruncatedDistribution(Normal(0, 1), 0, Inf))) == Float64
        @test realtype(typeof(TruncatedDistribution(Normal(0, 1), -Inf, 0))) == Float64
        @test realtype(typeof(TruncatedDistribution(Normal(2,3), 2, Inf))) == Float64
        @test realtype(typeof(TruncatedDistribution(Normal(2,3), 2, 3))) == Int64
        @test realtype(typeof(TruncatedDistribution(Normal(2,3), 2, 3.0))) == Float64
        @test realtype(typeof(TruncatedDistribution(Normal(2f0,3f0), 2, 3.0))) == Float64
        @test realtype(typeof(TruncatedDistribution(Normal(2f0,3f0), 2, 3))) == Float32
    end

    @testset "normalization_constant" begin
        @test normalization_constant(TruncatedDistribution(Normal(0, 1), -Inf, Inf)) ≈ 1
        @test normalization_constant(TruncatedDistribution(Normal(0, 1), -Inf, 0)) ≈ 0.5
        @test normalization_constant(TruncatedDistribution(Normal(2,3), 2, Inf)) ≈ 0.5
        @test normalization_constant(TruncatedDistribution(Poisson(1), 0, 3)) ≈ sum(pmf.(Ref(Poisson(1)), 0:3))
        @test normalization_constant(TruncatedDistribution(Poisson(1), 1, 3)) ≈ sum(pmf.(Ref(Poisson(1)), 1:3))
    end

    @testset "lognormalization_constant" begin
        @test lognormalization_constant(TruncatedDistribution(Normal(0, 1), -Inf, Inf)) ≈ 0
        @test lognormalization_constant(TruncatedDistribution(Normal(0, 1), -Inf, 0)) ≈ log(0.5)
        @test lognormalization_constant(TruncatedDistribution(Normal(2,3), 2, Inf)) ≈ log(0.5)
        @test lognormalization_constant(TruncatedDistribution(Poisson(1), 0, 3)) ≈ log(sum(pmf.(Ref(Poisson(1)), 0:3)))
        @test lognormalization_constant(TruncatedDistribution(Poisson(1), 1, 3)) ≈ log(sum(pmf.(Ref(Poisson(1)), 1:3)))
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

    @testset "pmf" begin
        @test pmf(TruncatedDistribution(Poisson(1), 0, 3), -1) ≈ 0
    end
    
    @testset "cdf" begin
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), -1) ≈ 0
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 0) ≈ 0
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 2) ≈ 1 - 2 * cdf(Normal(0, 1), -2)
        @test cdf(TruncatedDistribution(Normal(0, 1), 0, Inf), Inf) ≈ 1
        @test cdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), 0) ≈ 1
        @test cdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), 1) ≈ 1
    end

    @testset "logcdf" begin
        @test logcdf(TruncatedDistribution(Normal(0, 1), 0, Inf), -1) ≈ -Inf
        @test logcdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 0) ≈ -Inf
        @test logcdf(TruncatedDistribution(Normal(0, 1), 0, Inf), 2) ≈ log(1 - 2 * cdf(Normal(0, 1), -2))
        @test logcdf(TruncatedDistribution(Normal(0, 1), 0, Inf), Inf) ≈ 0
        @test logcdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), 0) ≈ 0
        @test logcdf(TruncatedDistribution(Normal(0, 1), -Inf, 0), 1) ≈ 0
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

    @testset "truncate_to_quantiles" begin
        @test truncate_to_quantiles(Normal(0,1), 0, 1) == TruncatedDistribution(Normal(0, 1), -Inf, Inf)
    end

    @testset "expand_truncation_to_ints" begin
        @test expand_truncation_to_ints(TruncatedDistribution(Normal(0, 1), 0, 5.4)) == TruncatedDistribution(Normal(0, 1), 0, 6)
        @test expand_truncation_to_ints(TruncatedDistribution(Normal(0, 1), -5.4, 0)) == TruncatedDistribution(Normal(0, 1), -6, 0)
        @test expand_truncation_to_ints(TruncatedDistribution(Normal(0, 1), -5.4, 5.4)) == TruncatedDistribution(Normal(0, 1), -6, 6)
    end

end