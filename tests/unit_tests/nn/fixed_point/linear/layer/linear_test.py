from dataclasses import dataclass
from unittest import TestCase, main

import torch
from torch import Tensor, cat, tensor, testing, zeros
from torch.nn import BatchNorm1d, Linear, Sequential

from elasticai.creator.nn import Sequential as SequentialCreator
from elasticai.creator.nn.fixed_point import (
    BatchNormedLinear as BatchNormedLinearCreator,
)
from elasticai.creator.nn.fixed_point import (
    Linear as LinearCreator,
)


@dataclass
class SettingsTest:
    data_repeat_num: int = 3
    size_input_layer: int = 3
    size_output_layer: int = 1
    quant_width_total: int = 12
    quant_width_frac: int = 8


RecommendedSettings = SettingsTest()


def generate_test_data(settings: SettingsTest = RecommendedSettings) -> Tensor:
    outputs = zeros((1, settings.size_input_layer)).repeat(settings.data_repeat_num, 1)
    for idx in range(settings.data_repeat_num):
        outputs = cat((outputs, tensor([[-1.5, 0.5, 1.5]])), dim=0)
        outputs = cat((outputs, tensor([[1.5, 0.5, -1.5]])), dim=0)
    return outputs


def generate_torch_linear(settings: SettingsTest = RecommendedSettings) -> list:
    linear_layer = Linear(
        settings.size_input_layer, settings.size_output_layer, bias=True
    )
    linear_layer.bias.data = tensor(0.5)
    linear_layer.weight.data = tensor([[0.5, 1.0, -0.5]])
    return [linear_layer]


def generate_creator_linear(settings: SettingsTest = RecommendedSettings) -> list:
    linear_layer = LinearCreator(
        in_features=settings.size_input_layer,
        out_features=settings.size_output_layer,
        total_bits=settings.quant_width_total,
        frac_bits=settings.quant_width_frac,
        bias=True,
    )
    linear_layer.bias.data = tensor(0.5)
    linear_layer.weight.data = tensor([[0.5, 1.0, -0.5]])
    return [linear_layer]


def generate_torch_batchlinear(settings: SettingsTest = RecommendedSettings) -> list:
    linear_layer = generate_torch_linear(settings)

    batch1d_layer = BatchNorm1d(settings.size_output_layer)
    batch1d_layer.bias.data = tensor([0.25])
    batch1d_layer.weight.data = tensor([0.75])
    linear_layer.append(batch1d_layer)
    return linear_layer


def generate_creator_batchlinear(settings: SettingsTest = RecommendedSettings) -> list:
    batch1d_layer = BatchNormedLinearCreator(
        in_features=settings.size_input_layer,
        out_features=settings.size_output_layer,
        total_bits=settings.quant_width_total,
        frac_bits=settings.quant_width_frac,
        bias=True,
        bn_affine=True,
    )
    batch1d_layer.lin_bias.data = tensor([[0.5]])
    batch1d_layer.lin_weight.data = tensor([[0.5, 1.0, -0.5]])

    batch1d_layer.bn_bias.data = tensor([0.25])
    batch1d_layer.bn_weight.data = tensor([0.75])
    return [batch1d_layer]


def build_torch_sequential(nn_layer: list, do_training: bool = False) -> Sequential:
    model_build = Sequential()
    for layer in nn_layer:
        model_build.append(layer)

    if do_training:
        model_build.train()
    else:
        model_build.eval()
    return model_build


def build_creator_sequential(
    nn_layer: list, do_training: bool = False
) -> SequentialCreator:
    model_build = SequentialCreator()
    for layer in nn_layer:
        model_build.append(layer)

    if do_training:
        model_build.train()
    else:
        model_build.eval()
    return model_build


class TestCreatorLinear(TestCase):
    model_torch = build_torch_sequential(generate_torch_linear())
    model_creator = build_creator_sequential(generate_creator_linear())
    data_in = generate_test_data()

    def test_data_input_size(self):
        testing.assert_close(self.data_in.size(1), RecommendedSettings.size_input_layer)

    def test_data_output_size_torch(self):
        testing.assert_close(
            self.model_torch(self.data_in).size(1),
            RecommendedSettings.size_output_layer,
        )

    def test_data_output_size_creator(self):
        testing.assert_close(
            self.model_creator(self.data_in).size(1),
            RecommendedSettings.size_output_layer,
        )

    def test_creator_linear(self):
        data_out_torch = self.model_torch(self.data_in)
        data_out_creator = self.model_creator(self.data_in)
        testing.assert_close(
            data_out_torch, data_out_creator, atol=1 / (2**8), rtol=1 / (2**8)
        )


def test_inference_of_multidimensional_data() -> None:
    linear = LinearCreator(
        total_bits=16, frac_bits=8, in_features=3, out_features=2, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data)

    inputs = torch.tensor([1.0, 2.0, 3.0])
    expected = [6.0, 6.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_overflow_behaviour() -> None:
    linear = LinearCreator(
        total_bits=4, frac_bits=1, in_features=2, out_features=1, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data) * 2

    inputs = torch.tensor([2.5, -1.0])
    expected = [3.0]  # quantize(2.5 * 2 - 1.0 * 2)
    actual = linear(inputs).tolist()

    assert expected == actual


def test_underflow_behaviour() -> None:
    linear = LinearCreator(
        total_bits=4, frac_bits=1, in_features=1, out_features=1, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data) * 0.5

    inputs = torch.tensor([0.5])
    expected = [0.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_bias_addition() -> None:
    linear = LinearCreator(
        total_bits=16, frac_bits=8, in_features=1, out_features=1, bias=True
    )
    linear.weight.data = torch.ones_like(linear.weight.data)
    linear.bias.data = torch.ones_like(linear.bias.data) * 2

    inputs = torch.tensor([3.0])
    expected = [5.0]
    actual = linear(inputs).tolist()

    assert expected == actual


if __name__ == "__main__":
    main()
