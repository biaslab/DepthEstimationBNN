@testitem "Skip layer" begin

    using UnboundedBNN: Skip
    import UnboundedBNN: KL_loss

    struct IdentityLayer end

    function (l::IdentityLayer)(x_mean, x_var)
        return x_mean, x_var
    end

    KL_loss(::IdentityLayer) = 0.0

    @test Skip(IdentityLayer())(5.0, 6.0) == (10.0, 12.0)
    
    m = randn(1000)
    v = rand(1000)
    @test Skip(IdentityLayer())(m, v) == (2*m, 2*v)

    @test KL_loss(Skip(IdentityLayer())) == 0.0

end