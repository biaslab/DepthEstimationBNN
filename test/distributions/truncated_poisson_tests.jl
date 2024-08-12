@testitem "Truncated Poisson distribution" begin

    using UnboundedBNN: TruncatedPoisson, get_iλ, get_λ, mean, var, pdf, KL_loss, softplus, invsoftplus, min_support, max_support, normalization_constant, support

    @test get_iλ(TruncatedPoisson([1.0])) == invsoftplus(1.0)
    @test get_iλ(TruncatedPoisson(1.0; warn=false)) == invsoftplus(1.0)

    @test get_λ(TruncatedPoisson([1.0])) == 1.0
    @test get_λ(TruncatedPoisson(1.0; warn=false)) == 1.0
    
    @test_logs (:warn,"TruncatedPoisson initialized with scalar. Zygote will not be able to update scalars. Use arrays instead, as in TruncatedPoisson([λ]).") TruncatedPoisson(1.0) 

    @test all(min_support.(abs.(10*randn(100))) .≥ 0)
    @test all(max_support.(abs.(10*randn(100))) .≥ 0)
    for n in abs.(10*randn(100))
        @test min_support(n) ≤ max_support(n)
        @test min_support(TruncatedPoisson(n; warn=false)) ≤ max_support(TruncatedPoisson(n; warn=false))
        @test min_support(TruncatedPoisson([n])) ≤ max_support(TruncatedPoisson([n]))
        @test min_support(TruncatedPoisson([n])) ≤ n
        @test max_support(TruncatedPoisson([n])) ≥ n
    end

    for n in 1.0:10.0
        @test 0 ≤ normalization_constant(TruncatedPoisson([n])) ≤ 1
    end

    for n in 1.0:10.0
        for x in min_support(n):max_support(n)
            @test 0 ≤ pdf(TruncatedPoisson([n]), x) ≤ 1
            @test pdf(TruncatedPoisson(n; warn=false), x) ≥ pdf(Poisson(n), x)
        end
        @test pdf(TruncatedPoisson([n]), min_support(n)-1) ≈ 0
        @test pdf(TruncatedPoisson([n]), max_support(n)+1) ≈ 0
        mapreduce(x -> pdf(TruncatedPoisson([n]), x), +, min_support(n):max_support(n)) ≈ 1
    end

    for n = 1.0:10.0
        p = TruncatedPoisson(n; warn=false)
        q = Poisson(n)
        pdf_p = map(x -> pdf(p, x), support(p))
        pdf_q = map(x -> pdf(q, x), support(p))
        KL = sum(pdf_p .* log.(pdf_p ./ pdf_q))
        @test KL_loss(p, q) ≈ KL
    end
end