export LeakyReLU

Base.@kwdef struct LeakyReLU 
    l::Float32 = 0.1f0
end

(l::LeakyReLU)(x) = relu(x; l=l.l)