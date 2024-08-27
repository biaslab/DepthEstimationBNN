@testitem "Leaky ReLU" begin

    using UnboundedBNN: LeakyReLU, relu
    
    A = randn(100, 200)
    layer = LeakyReLU()
    @test all(layer(A) .== relu(A; l=0.1f0))

end