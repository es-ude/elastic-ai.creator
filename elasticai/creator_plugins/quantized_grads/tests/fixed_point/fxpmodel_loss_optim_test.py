import pytest
import torch
from torch.nn import Sequential
from torch.optim import SGD

from elasticai.creator_plugins.quantized_grads.base_modules import Linear
from elasticai.creator_plugins.quantized_grads.fixed_point import (
    FixedPointConfigV2,
    QuantizeForwHTEBackwHTE,
    QuantizeParamToFixedPointHTE,
)
from elasticai.creator_plugins.quantized_grads.quantized_optim import (
    get_quantized_optimizer,
)


@pytest.mark.slow
class Test1:
    def test_pred(self, pred: torch.Tensor) -> None:
        expected_pred = torch.Tensor([0.5, 2])
        assert torch.equal(pred, expected_pred)

    def test_loss(self, loss):
        expected_loss = 0.6250
        assert loss == expected_loss

    def test_grad_weight(self, model, loss):
        expected_grad_weight = torch.Tensor([[0, 0, 0], [0, 1, 2]])
        actual_grad_weight = model[0].weight.grad
        assert torch.equal(actual_grad_weight, expected_grad_weight)

    def test_grad_bias(self, model, loss):
        expected_grad_bias = torch.Tensor([0, 1])
        actual_grad_bias = model[0].bias.grad
        assert torch.equal(actual_grad_bias, expected_grad_bias)

    def test_new_weight(self, model, optimizer_step):
        expected_new_weight = torch.Tensor([[0, 0.125, 0.25], [0.375, 0.375, 0.375]])
        actual_new_weight = model[0].weight
        assert torch.equal(actual_new_weight, expected_new_weight)

    def test_new_bias(self, model, optimizer_step):
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
        params_conf_frac_bits = 3
        params_conf = FixedPointConfigV2(
            total_bits=params_conf_total_bits, frac_bits=params_conf_frac_bits
        )

        forward_conf_total_bits = 8
        forward_conf_frac_bits = 2
        forward_conf = FixedPointConfigV2(
            total_bits=forward_conf_total_bits,
            frac_bits=forward_conf_frac_bits,
        )

        grad_conf_total_bits = 8
        grad_conf_frac_bits = 0
        backward_conf = FixedPointConfigV2(
            total_bits=grad_conf_total_bits,
            frac_bits=grad_conf_frac_bits,
        )

        nn = Sequential(
            Linear(
                math_ops=QuantizeForwHTEBackwHTE(
                    forward_conf=forward_conf, backward_conf=backward_conf
                ),
                in_features=in_features,
                out_features=out_features,
                weight_quantization=QuantizeParamToFixedPointHTE(params_conf),
                bias_quantization=QuantizeParamToFixedPointHTE(params_conf),
                bias=True,
            )
        )
        nn[0].weight = torch.reshape(
            input=torch.arange(
                start=0,
                end=(out_features * in_features) / (2**params_conf_frac_bits),
                step=1 / (2**params_conf_frac_bits),
                requires_grad=True,
                dtype=torch.float32,
            ),
            shape=nn[0].weight.shape,
        )
        nn[0].bias = torch.reshape(
            input=torch.arange(
                start=0,
                end=out_features / (2**params_conf_frac_bits),
                step=1 / (2**params_conf_frac_bits),
                dtype=torch.float32,
            ),
            shape=nn[0].bias.shape,
        )
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
    def optimizer(self, model: Sequential, loss: torch.Tensor) -> torch.optim.Optimizer:
        QSGD = get_quantized_optimizer(SGD)
        optimizer = QSGD(
            model=model,
            params=model.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        return optimizer

    @pytest.fixture(scope="class")
    def optimizer_step(self, optimizer: SGD, loss: torch.Tensor) -> None:
        optimizer.step()


@pytest.mark.slow
class Test2:
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

    def test_new_weight(self, model, optimizer_step):
        expected_new_weight = torch.Tensor([[0, 0, 0.25], [0.75, 0.75, 0.75]])
        actual_new_weight = model[0].weight
        assert torch.equal(actual_new_weight, expected_new_weight)

    def test_new_bias(self, model, optimizer_step):
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

        nn = Sequential(
            Linear(
                math_ops=QuantizeForwHTEBackwHTE(
                    forward_conf=forward_conf, backward_conf=backward_conf
                ),
                in_features=in_features,
                out_features=out_features,
                weight_quantization=QuantizeParamToFixedPointHTE(params_conf),
                bias_quantization=QuantizeParamToFixedPointHTE(params_conf),
                bias=True,
            )
        )
        nn[0].weight = torch.reshape(
            input=torch.arange(
                start=0,
                end=(out_features * in_features) / (2**params_conf_frac_bits),
                step=1 / (2**params_conf_frac_bits),
                requires_grad=True,
                dtype=torch.float32,
            ),
            shape=nn[0].weight.shape,
        )
        nn[0].bias = torch.reshape(
            input=torch.arange(
                start=0,
                end=out_features / (2**params_conf_frac_bits),
                step=1 / (2**params_conf_frac_bits),
                dtype=torch.float32,
            ),
            shape=nn[0].bias.shape,
        )
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
    def optimizer(self, model: Sequential, loss: torch.Tensor) -> torch.optim.Optimizer:
        QSGD = get_quantized_optimizer(SGD)
        optimizer = QSGD(
            model=model,
            params=model.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        return optimizer

    @pytest.fixture(scope="class")
    def optimizer_step(self, optimizer: SGD, loss: torch.Tensor) -> None:
        optimizer.step()
