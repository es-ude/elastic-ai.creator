import pytest

import torch
from torch import Tensor
from torch.nn import Sequential
from torch.optim import SGD

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules import Linear

from elasticai.creator_plugins.quantized_grads.linear_quantization import (
    LinearQuantizationConfig,
    IntQuantizationConfig,
    QuantizeForwHTEBackwHTE,
    QuantizeParamToLinearQuantizationHTE,
    quantize_linear_asym_hte_fake, quantize_linear_asym_hte, QuantizeTensorToIntHTE, dequantize_linear
)
from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import QuantizeParamToIntHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.quantize_to_int_with_linear_quantization_style import \
    quantize_to_int_hte, quantize_to_int_hte_fake

from elasticai.creator_plugins.quantized_grads.quantized_optim import get_quantized_optimizer

def get_linear_quantized_tensor(t: Tensor, num_bits: int) -> Tensor:
    return quantize_linear_asym_hte_fake(t, Tensor([0]), Tensor([2 ** num_bits - 1]))

def get_int_quantized_tensor(t: Tensor, num_bits: int) -> Tensor:
    return quantize_to_int_hte_fake(t, Tensor([-2**(num_bits-1)]), Tensor([2**(num_bits-1)-1]))

torch.set_printoptions(precision=16)
@pytest.mark.slow
class TestLinearQuantizationTraining:
    def test_pred(self, pred: Tensor) -> None:
        """
        [13054, = [6528+ 6528 +0-2,=  [[  -102,  -51., 0] @ [ -64.,   -128., 127.] + [-2, 1]
        3112] =  -3264-13056+19431+1]   [51, 102, 153]]
        """
        expected_pred = Tensor([1.00037205219268798828125, 0.2388948500156402587890625])
        assert torch.equal(pred, expected_pred)

    def test_loss(self, loss: Tensor) -> None:
        print(loss)
        expected_loss = Tensor([3.5514895915985107421875])
        assert torch.equal(Tensor([loss]), expected_loss)

    def test_grad_weight(self, loss: Tensor, model: Sequential) -> None:
        expected_grad_weight = Tensor([])
        actual_grad_weight = model[0].weight.grad
        g, g_scale, g_zero_point = quantize_linear_asym_hte(actual_grad_weight, Tensor([0]), Tensor([2 ** (8) - 1]))
        print(f"{actual_grad_weight=}")
        print(f"{g=}, {g_scale=}, {g_zero_point=}")
        print(f"{dequantize_linear(g, g_scale, g_zero_point)=}")
        assert torch.equal(actual_grad_weight, expected_grad_weight)

    def test_grad_bias(self, loss: Tensor, model: Sequential) -> None:
        expected_grad_bias = Tensor([])
        actual_grad_bias = model[0].bias.grad
        print(f"{actual_grad_bias=}")
        assert torch.equal(actual_grad_bias, expected_grad_bias)

    def test_new_weight(self, model: Sequential, optimizer_step: None) -> None:
        expected_new_weight = Tensor([])
        actual_new_weight = model[0].weight
        assert torch.equal(actual_new_weight, expected_new_weight)

    def test_new_bias(self, model: Sequential, optimizer_step: None) -> None:
        expected_new_bias = Tensor([])
        actual_new_bias = model[0].bias
        assert torch.equal(actual_new_bias, expected_new_bias)

    @pytest.fixture(scope="class")
    def in_features(self) -> int:
        return 3

    @pytest.fixture(scope="class")
    def out_features(self) -> int:
        return 2

    @pytest.fixture(scope="class")
    def params_num_bits(self) -> int:
        return 8

    @pytest.fixture(scope="class")
    def output_num_bits(self) -> int:
        return 8

    @pytest.fixture(scope="class")  # ensures expensive setup is performed only once
    def model(self, in_features: int, params_num_bits: int, output_num_bits: int, out_features: int) -> Sequential:
        print()
        params_conf_total_bits = params_num_bits
        weight_conf = LinearQuantizationConfig(num_bits=params_conf_total_bits)

        bias_conf = IntQuantizationConfig(num_bits=params_conf_total_bits)

        forward_conf_total_bits = output_num_bits
        forward_conf = LinearQuantizationConfig(num_bits=forward_conf_total_bits)

        grad_conf_total_bits = 8
        backward_conf = LinearQuantizationConfig(num_bits=grad_conf_total_bits)

        nn = Sequential(
            Linear(
                output_quantization=QuantizeForwHTEBackwHTE(
                    forward_conf=forward_conf, backward_conf=backward_conf
                ),
                in_features=in_features,
                out_features=out_features,
                weight_quantization=QuantizeParamToLinearQuantizationHTE(weight_conf),
                bias_quantization=QuantizeParamToIntHTE(bias_conf),
                bias=True,
            )
        )

        weight = get_linear_quantized_tensor(Tensor([-0.5, -0.25, 0., 0.25, 0.5, 0.75]), params_conf_total_bits)
        bias = get_int_quantized_tensor(Tensor([-2., 1.]), params_conf_total_bits)
        print(f"{bias=}")


        nn[0].weight = torch.reshape(
            input=weight,
            shape=nn[0].weight.shape,
        )
        nn[0].bias = torch.reshape(
            input=bias,
            shape=nn[0].bias.shape,
        )
        #print(f"{nn[0].weight=}")
        #print(f"{nn[0].bias=}")

        print()
        w_, w_scale, w_zero_point = quantize_linear_asym_hte(nn[0].weight, Tensor([0.]), Tensor([2 ** params_num_bits - 1]))
        print(f"{w_=}, {w_scale=}, {w_zero_point=}")

        b_, b_scale, b_zero_point = quantize_to_int_hte(nn[0].bias, Tensor([-2**(params_num_bits-1)]), Tensor([2**(params_num_bits-1)-1]))
        print(f"{nn[0].bias=}")
        print(f"{b_=}, {b_scale=}, {b_zero_point=}")

        return nn

    @pytest.fixture(scope="class")
    def input(self, in_features: int, params_num_bits: int) -> Tensor:
        x = get_linear_quantized_tensor(Tensor([-1., -2., 2.]), params_num_bits)
        x_, x_scale, x_zero_point = quantize_linear_asym_hte(x, Tensor([0.]), Tensor([2 ** params_num_bits - 1]))
        print(f"{x_=}, {x_scale=}, {x_zero_point=}")
        return x

    @pytest.fixture(scope="class")
    def pred(self, input: Tensor, model: Sequential) -> Tensor:
        return model(input)

    @pytest.fixture(scope="class")
    def label(self, out_features: int, output_num_bits: int) -> Tensor:
        return get_linear_quantized_tensor(Tensor([-1., 2.]), output_num_bits)

    @pytest.fixture(scope="class")
    def loss(self, pred: Tensor, label: Tensor) -> Tensor:
        loss_fn = torch.nn.MSELoss()
        _loss = loss_fn(pred, label)
        _loss.backward()
        return _loss

    @pytest.fixture(scope="class")
    def optimizer(self, model: Sequential) -> torch.optim.Optimizer:
        QSGD = get_quantized_optimizer(SGD)
        optimizer = QSGD(model=model, params=model.parameters(), lr=0.01, momentum=0.9, buffer_quantizations={"momentum_buffer": lambda x:x})
        return optimizer

    @pytest.fixture(scope="class")
    def optimizer_step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.step()

