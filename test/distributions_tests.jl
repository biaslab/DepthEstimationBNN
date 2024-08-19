@testitem "Distributions" begin
    
    using UnboundedBNN
    using UnboundedBNN: SupportDistribution, ShiftedHeaviside, Distribution

    @testset "Heaviside truncation" begin

        @test H() * Normal(0, 1) == TruncatedDistribution(Normal(0,1), 0.0, Inf)
        @test Normal(0, 1) * H() == TruncatedDistribution(Normal(0,1), 0.0, Inf)
        @test ShiftedHeaviside(5) * Normal(0, 1) == TruncatedDistribution(Normal(0,1), 5.0, Inf)
        @test Normal(0, 1) * ShiftedHeaviside(5) == TruncatedDistribution(Normal(0,1), 5.0, Inf)
    
    end

    @testset "Support distribution" begin

        @test SupportDistribution(Normal(0, 1)) isa SupportDistribution{<:Distribution, -Inf, Inf}
        @test SupportDistribution(TruncatedDistribution(Normal(0,1), 0, 5)) isa SupportDistribution{<:Distribution, 0, 5}

    end

end