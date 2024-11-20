import torch
from torch.nn import CrossEntropyLoss

from elasticai.creator.nn.training.fixed_point import FixedPointConfigV2, Linear

if __name__ == "__main__":
    in_features = 3
    out_features = 2

    total_bits_params = 8
    frac_bits_params = 6

    total_bits_forward = 8
    frac_bits_forward = 4
    stochastic_forward = True

    total_bits_backward = 8
    frac_bits_backward = 4
    stochastic_backward = True

    forward_conf = FixedPointConfigV2(
        total_bits=total_bits_forward,
        frac_bits=frac_bits_forward,
        stochastic_rounding=stochastic_forward,
    )
    backward_conf = FixedPointConfigV2(
        total_bits=total_bits_backward,
        frac_bits=frac_bits_backward,
        stochastic_rounding=stochastic_backward,
    )

    my_layer = Linear(in_features, out_features, forward_conf, backward_conf)

    x = torch.ones((in_features))
    y = torch.ones((out_features))

    pred = my_layer(x)

    loss = CrossEntropyLoss(y, pred)

    loss.backward()

    op
    print(pred)
