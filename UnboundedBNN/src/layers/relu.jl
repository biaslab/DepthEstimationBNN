export ReLU

struct ReLU end

function (l::ReLU)(x_mean, x_var)
    return relu(x_mean, x_var)
end

function relu(m, v)
    s = sqrt.(v)
    is = 1 ./ s
    alpha = - m .* is
    Z = 1 .- normcdf.(alpha)
    # rectified normal distribution
    #https://math.stackexchange.com/questions/1963292/expectation-and-variance-of-gaussian-going-through-rectified-linear-or-sigmoid-f
    m_new = Z .* m + s .* normpdf.(alpha)
    v_new = (s.^2 + m.^2) .* Z + m .* s .* normpdf.(alpha) - m_new.^2
    return m_new, v_new
end