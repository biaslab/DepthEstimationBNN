@testitem "math" begin

    using UnboundedBNN: H, ShiftedHeaviside, sigmoid, invsoftplus, softplus
    
    @test H() == ShiftedHeaviside(0)
    @test sigmoid(0) == 0.5
    @test invsoftplus(softplus(1)) == 1
    @test softplus(invsoftplus(1)) == 1
    
end