import pytest
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


class Test:
    def test_pred(self, pred: torch.Tensor) -> None:
        expected_pred = torch.Tensor([1.25, 3.75])
        assert torch.equal(pred, expected_pred)

    def test_loss(self, loss):
        expected_loss = 4.5625
        assert loss == expected_loss

    def test_grad_weight(self, model, loss):
        expected_grad_weight = torch.Tensor([[0, 1.25, 2.5], [0, 2.75, 5.5]])
        actual_grad_weight = model[0].weight.grad
        assert torch.equal(actual_grad_weight, expected_grad_weight)

    def test_grad_bias(self, model, loss):
        expected_grad_bias = torch.Tensor([1.25, 2.75])
        actual_grad_bias = model[0].bias.grad
        assert torch.equal(actual_grad_bias, expected_grad_bias)

    def test_new_weight(self, model, optimizer):
        expected_new_weight = torch.Tensor([[0, 0, 0.25], [0.75, 0.75, 0.75]])
        actual_new_weight = model[0].weight
        assert torch.equal(actual_new_weight, expected_new_weight)

    def test_new_bias(self, model, optimizer):
        expected_new_bias = torch.Tensor([0, 0])
        actual_new_bias = model[0].bias
        assert torch.equal(actual_new_bias, expected_new_bias)

    @pytest.fixture(scope="class")
    def in_features(self) -> int:
        in_features = 3
        return in_features

    @pytest.fixture(scope="class")
    def out_features(self) -> int:
        out_features = 2
        return out_features

    @pytest.fixture(scope="class")  # ensures expensive setup is performed only once
    def model(self, in_features: int, out_features: int) -> Sequential:
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

        def round_params(params: torch.Tensor) -> None:
            quantize_to_fxp_hte_(params, params_conf)

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
        print(f"{nn[0].weight=}")
        print(f"{nn[0].bias=}")
        return nn

    @pytest.fixture(scope="class")
    def input(self, in_features: int) -> torch.Tensor:
        return torch.arange(start=0, end=in_features, dtype=torch.float32)

    @pytest.fixture(scope="class")
    def pred(self, model: Sequential, input: torch.Tensor) -> torch.Tensor:
        return model(input)

    @pytest.fixture(scope="class")
    def output(self, out_features: int) -> torch.Tensor:
        return torch.arange(start=0, end=out_features, dtype=torch.float32)

    @pytest.fixture(scope="class")
    def loss(self, output: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        loss_fn = torch.nn.MSELoss()
        _loss = loss_fn(pred, output)
        _loss.backward()
        return _loss

    @pytest.fixture(scope="class")
    def optimizer(self, model: Sequential, loss: torch.Tensor) -> QuantizedSGD:
        optimizer = QuantizedSGD(
            model,
            lr=0.1,
            momentum=0,
        )
        optimizer.step()
        return optimizer
