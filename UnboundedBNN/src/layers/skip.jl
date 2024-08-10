export Skip

struct Skip{ L }
    layer :: L
end
@functor Skip

function (l::Skip)(x_mean, x_var)
    y_mean, y_var = l.layer(x_mean, x_var)
    return y_mean + x_mean, y_var + x_var
end

KL_loss(s::Skip) = KL_loss(s.layer)