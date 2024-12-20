import torch
from torch.nn import Sequential

from elasticai.creator.nn.quantized_grads.base_modules import Linear
from elasticai.creator.nn.quantized_grads.fixed_point import (
    FixedPointConfigV2,
    MathOperationsForwBackwHTE,
)
from elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point import (
    quantize_to_fxp_hte_,
)
from elasticai.creator.nn.quantized_grads.quantized_sgd import QuantizedSGD


def test_layer_autograd_optimizer():
    params_conf_total_bits = 8
    params_conf_frac_bits = 2
    params_conf = FixedPointConfigV2(
        total_bits=params_conf_total_bits, frac_bits=params_conf_frac_bits
    )

    forward_conf_total_bits = 8
    forward_conf_frac_bits = 4
    forward_conf = FixedPointConfigV2(
        total_bits=forward_conf_total_bits,
        frac_bits=forward_conf_frac_bits,
    )

    grad_conf_total_bits = 8
    grad_conf_frac_bits = 4
    backward_conf = FixedPointConfigV2(
        total_bits=grad_conf_total_bits,
        frac_bits=grad_conf_frac_bits,
    )

    in_features = 3
    out_features = 2

    def round_params(params: torch.Tensor) -> torch.Tensor:
        return quantize_to_fxp_hte_(params, params_conf)

    nn = Sequential(
        Linear(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperationsForwBackwHTE(
                forward=forward_conf, backward=backward_conf
            ),
            weight_quantization=round_params,
            bias_quantization=round_params,
            bias=True,
        )
    )
    nn[0].weight = torch.nn.Parameter(
        torch.reshape(
            input=torch.arange(
                start=0,
                end=(out_features * in_features) / (2**params_conf_frac_bits),
                step=1 / (2**params_conf_frac_bits),
                requires_grad=True,
                dtype=torch.float32,
            ),
            shape=nn[0].weight.shape,
        )
    )
    nn[0].bias = torch.nn.Parameter(
        torch.reshape(
            input=torch.arange(
                start=0,
                end=out_features / (2**params_conf_frac_bits),
                step=1 / (2**params_conf_frac_bits),
                dtype=torch.float32,
            ),
            shape=nn[0].bias.shape,
        )
    )
    optimizer = QuantizedSGD(
        nn,
        lr=0.1,
        momentum=0,
    )

    loss_fn = torch.nn.MSELoss()

    x = torch.arange(start=0, end=in_features, dtype=torch.float32)
    y = torch.arange(start=0, end=out_features, dtype=torch.float32)

    expected_pred = torch.Tensor([1.25, 3.75])
    expected_loss = 4.5625
    expected_grad_weight = torch.Tensor([[0, 1.25, 2.5], [0, 2.75, 5.5]])
    expected_grad_bias = torch.Tensor([1.25, 2.75])
    expected_new_weight = torch.Tensor([[0, 0, 0.25], [0.75, 0.75, 0.75]])
    expected_new_bias = torch.Tensor([0, 0])

    actual_pred = nn(x)
    assert torch.equal(actual_pred, expected_pred)

    loss = loss_fn(actual_pred, y)
    actual_loss = loss.item()
    assert actual_loss == expected_loss

    loss.backward()

    actual_grad_weight = nn[0].weight.grad
    actual_grad_bias = nn[0].bias.grad
    assert torch.equal(actual_grad_weight, expected_grad_weight)
    assert torch.equal(actual_grad_bias, expected_grad_bias)

    optimizer.step()

    actual_new_weight = nn[0].weight
    actual_new_bias = nn[0].bias
    assert torch.equal(actual_new_weight, expected_new_weight)
    assert torch.equal(actual_new_bias, expected_new_bias)
