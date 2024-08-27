@testitem "Chain" begin
    
    using UnboundedBNN: Chain, LinearBBB, ReLU

    linear1 = LinearBBB(2 => 3; initializer=(1,0), eps=0)
    relu1 = ReLU()
    linear2 = LinearBBB(3 => 2; initializer=(1,0), eps=0)
    relu2 = ReLU()

    A = randn(2, 100)
    @test Chain(linear1, relu1)(A) ≈ relu1(linear1(A))
    @test Chain(linear1, relu1, linear2, relu2)(A) ≈ relu2(linear2(relu1(linear1(A))))

    @test KL_loss(Chain(LinearBBB(2=>2; initializer=(0,1), eps=0), relu1)) ≈ 0.0

end