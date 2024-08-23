@testitem "Spiral dataset" begin

    x, y = generate_spiral(1000, 1)
    @test size(x) == (2, 1000)
    @test size(y) == (1000,)
    @test eltype(x) == Float32

end