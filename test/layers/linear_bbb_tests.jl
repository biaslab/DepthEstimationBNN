@testitem "Linear layer using Bayes-by-backprop" begin
    
    using UnboundedBNN: LinearBBB
    
    @testset "forward" begin
        
        m = randn(Float32, 100, 50)
        out = LinearBBB(100 => 100)(m)
        @test size(out) == (100, 50)

    end

    @testset "KL" begin
        @test KL_loss(LinearBBB(100 => 100)) >= 0
        @test KL_loss(LinearBBB(100 => 100; eps = 1e-16)) ≈ 0
    end

end