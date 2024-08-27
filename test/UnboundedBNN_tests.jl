@testitem "UnboundedBNN" begin
    
    using UnboundedBNN: join_as_tuples, convert_to_tuples

    @testset "join_as_tuples" begin
        @test join_as_tuples([:a, :b, :c]) == :(tuple(a, b, c))
    end
    @testset "convert_to_tuples" begin
        @test convert_to_tuples([:a, :b, :c, :d]) == (:((a, b)), :((c, d)))
    end

end