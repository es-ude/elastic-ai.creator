import torch

from elasticai.creator.nn.training.fixed_point import (
    FixedPointConfigV2,
    Linear,
    QuantizedFxpSGD,
)


def test_layer_autograd_optimizer():
    weight_conf_total_bits = 8
    weight_conf_frac_bits = 2
    weight_conf_stochastic_rounding = False

    forward_conf_total_bits = 8
    forward_conf_frac_bits = 4
    forward_conf_stochastic_rounding = False

    grad_conf_total_bits = 8
    grad_conf_frac_bits = 4
    grad_conf_stochastic_rounding = False

    in_features = 3
    out_features = 2

    weight_conf = FixedPointConfigV2(
        total_bits=weight_conf_total_bits,
        frac_bits=weight_conf_frac_bits,
        stochastic_rounding=weight_conf_stochastic_rounding,
    )
    forward_conf = FixedPointConfigV2(
        total_bits=forward_conf_total_bits,
        frac_bits=forward_conf_frac_bits,
        stochastic_rounding=forward_conf_stochastic_rounding,
    )
    grad_conf = FixedPointConfigV2(
        total_bits=grad_conf_total_bits,
        frac_bits=grad_conf_frac_bits,
        stochastic_rounding=grad_conf_stochastic_rounding,
    )
    nn = Linear(
        in_features=in_features,
        out_features=out_features,
        param_fxp_conf=weight_conf,
        forward_fxp_conf=forward_conf,
        grad_fxp_conf=grad_conf,
    )
    nn.weight = torch.nn.Parameter(
        torch.reshape(
            input=torch.arange(
                start=0,
                end=(out_features * in_features) / (2**weight_conf_frac_bits),
                step=1 / (2**weight_conf_frac_bits),
                requires_grad=True,
                dtype=torch.float32,
            ),
            shape=nn.weight.shape,
        )
    )
    nn.bias = torch.nn.Parameter(
        torch.reshape(
            input=torch.arange(
                start=0,
                end=out_features / (2**weight_conf_frac_bits),
                step=1 / (2**weight_conf_frac_bits),
                dtype=torch.float32,
            ),
            shape=nn.bias.shape,
        )
    )
    optimizer = QuantizedFxpSGD(
        nn.parameters(),
        fxp_conf=weight_conf,
        save_quantization_error=False,
        lr=0.1,
        momentum=0,
        debug_print=True,
    )

    loss_fn = torch.nn.MSELoss()

    x = torch.arange(start=0, end=in_features, dtype=torch.float32)
    y = torch.arange(start=0, end=out_features, dtype=torch.float32)

    expected_pred = torch.Tensor([1.25, 3.75])
    expected_loss = 4.5625
    expected_grad_weight = torch.Tensor([[0, 1.25, 2.5], [0, 2.75, 5.5]])
    expected_grad_bias = torch.Tensor([1.25, 2.75])
    expected_new_weight = torch.Tensor([[0, 0.25, 0.25], [0.75, 0.75, 0.75]])
    expected_new_bias = torch.Tensor([0, 0])

    actual_pred = nn(x)
    loss = loss_fn(actual_pred, y)
    loss.backward()
    actual_loss = loss.item()

    actual_grad_weight = nn.weight.grad
    actual_grad_bias = nn.bias.grad

    optimizer.step()

    actual_new_weight = nn.weight
    actual_new_bias = nn.bias

    assert torch.equal(actual_pred, expected_pred)
    assert actual_loss == expected_loss
    assert torch.equal(actual_grad_weight, expected_grad_weight)
    assert torch.equal(actual_grad_bias, expected_grad_bias)
    assert torch.equal(actual_new_weight, expected_new_weight)
    assert torch.equal(actual_new_bias, expected_new_bias)
