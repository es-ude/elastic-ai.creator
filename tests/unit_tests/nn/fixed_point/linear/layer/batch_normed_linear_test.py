from dataclasses import dataclass
from unittest import TestCase, main

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


class TestCreatorBatchTraining(TestCase):
    model_torch = build_torch_sequential(
        generate_torch_batchlinear(), do_training=False
    )
    model_creator = build_creator_sequential(
        generate_creator_batchlinear(), do_training=False
    )
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

    def test_creator_batch(self):
        data_out_torch = self.model_torch(self.data_in)
        data_out_creator = self.model_creator(self.data_in)
        testing.assert_close(
            data_out_torch,
            data_out_creator,
            atol=1 / (2**RecommendedSettings.quant_width_frac),
            rtol=1 / (2**RecommendedSettings.quant_width_frac),
        )


class TestCreatorBatchEvaluation(TestCase):
    model_torch = build_torch_sequential(generate_torch_batchlinear(), do_training=True)
    model_creator = build_creator_sequential(
        generate_creator_batchlinear(), do_training=True
    )
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

    def test_creator_batch(self):
        data_out_torch = self.model_torch(self.data_in)
        data_out_creator = self.model_creator(self.data_in)
        testing.assert_close(
            data_out_torch,
            data_out_creator,
            atol=1 / (2**RecommendedSettings.quant_width_frac),
            rtol=1 / (2**RecommendedSettings.quant_width_frac),
        )


if __name__ == "__main__":
    main()
