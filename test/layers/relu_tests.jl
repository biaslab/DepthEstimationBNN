@testitem "ReLU layer" begin
    
    using UnboundedBNN: relu

    @testset "mean-var" begin
        
        m = randn(1000)
        v = rand(1000)
        m_out, v_out = relu(m, v)
        out = relu.(m, v)
        @test all(m_out .== first.(out))
        @test all(v_out .== last.(out))
        @test all(m_out .>= m .- 1e-10)
        @test all(m_out .>= 0)
        @test all(v_out .<= v .+ 1e-10)
        @test all(v_out .>= 0)

    end

    @testset "mean" begin
        
        m = randn(1000)
        out = relu(m)
        @test all(out .>= m)
        @test all(out .>= 0)

    end

end