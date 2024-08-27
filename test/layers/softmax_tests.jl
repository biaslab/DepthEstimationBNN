@testitem "Softmax" begin
    
    using UnboundedBNN: Softmax
    using StatsFuns: softmax
    
    A = randn(100, 200)
    layer = Softmax(200)
    @test all(exp.(layer(A)) .≈ softmax(A, dims = 1))

    @test sum(exp, layer(randn(1000))) ≈ 1.0
    @test sum(exp, layer(randn(1000, 100))) ≈ 100.0 

    @test all(layer(A, zeros(size(A)))[1] .≈ layer(A))
    @test layer(A, zeros(size(A)))[2] === nothing

end