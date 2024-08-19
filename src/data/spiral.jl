export generate_spiral

using Random: default_rng

function generate_spiral(N, ω; T=Float32, rng=default_rng())
    t = rand(rng, T, N)
    u = sqrt.(t)
    y = rand(rng, (-1, 1), N)
    x = zeros(T, 2, N)
    for n in 1:N
        tmp = randn(rng, T, 2)
        x[1, n] = y[n] * u[n] * cos(ω * u[n] * convert(T, pi) / 2) + convert(T, 0.02) * tmp[1]
        x[2, n] = y[n] * u[n] * sin(ω * u[n] * convert(T, pi) / 2) + convert(T, 0.02) * tmp[2]
    end
    return x, y
end