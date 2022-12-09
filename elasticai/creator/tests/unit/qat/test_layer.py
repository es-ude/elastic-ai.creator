import unittest
from functools import partial
from typing import Optional, Tuple
from unittest import SkipTest
from unittest.mock import patch

import torch
from torch import Tensor

from elasticai.creator.qat.blocks import BatchNormedActivatedConv1d
from elasticai.creator.qat.constraints import WeightClipper
from elasticai.creator.qat.layers import (
    QLSTM,
    Binarize,
    BinaryActivatedConv1d,
    BinarySplitConv,
    ChannelShuffle,
    MultilevelResidualBinarizationConv1d,
    QConv1d,
    QConv2d,
    QLinear,
    QLSTMCell,
    QuantizeTwoBit,
    ResidualQuantization,
    SplitConvolutionBase,
    Ternarize,
    TernaryConvolution1d,
    TernarySplitConv,
    TwoBitSplitConv,
)
from elasticai.creator.tests.unit.tensor_test_case import TensorTestCase


class MockQLSTMCell:
    def __init__(*args, **kwargs):
        pass

    # it is the forward function in a real LSTM cell
    def __call__(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None):
        return hx


class TernarizeTest(unittest.TestCase):
    def test_quantize_different_values(self) -> None:
        layer = Ternarize()
        out = layer(torch.tensor([1.0, -1.0, 0.0]))
        self.assertEqual(out.tolist(), [1.0, -1.0, 0.0])

    def test_does_not_require_grad_by_default(self) -> None:
        layer = Ternarize()
        out = layer(torch.tensor([2.0]))
        self.assertEqual(out.requires_grad, False)

    def test_widening_is_trainable_when_trainable_equals_true(self) -> None:
        layer = Ternarize(zero_window_width=0.5, trainable=True)
        out = layer(torch.tensor([2.0]))
        self.assertEqual(out.requires_grad, True)


class ResidualQuantizationTest(unittest.TestCase):
    def test_quantize_different_values(self) -> None:
        layer = ResidualQuantization()
        out = layer(torch.tensor([[1.0, -1.0, 0.0]]))
        self.assertEqual(out.tolist(), [[1.0, -1.0, 1.0, 1.0, -1.0, -1.0]])

    def test_backward_pass_requires_grad(self) -> None:
        layer = ResidualQuantization()
        out = layer(torch.tensor([[2.0]]))
        self.assertEqual(out.requires_grad, True)


class QuantizeTwoBitTest(unittest.TestCase):
    def test_call_with_0_3_factor_and_2_as_input(self) -> None:
        layer = QuantizeTwoBit(0.3)
        out = layer(torch.tensor([[2.0]]))
        self.assertEqual(out.tolist(), [[1, 1.0]])

    def test_factors_are_trainable(self) -> None:
        layer = QuantizeTwoBit(0.3)
        out = layer(torch.tensor([[2.0]]))
        self.assertEqual(out.requires_grad, True)


class QLinearTest(unittest.TestCase):
    def test_binarized_call_with_bias(self) -> None:
        layer = QLinear(in_features=1, out_features=2, quantizer=Binarize())
        layer.weight = torch.tensor([[2.0], [-2.0]])
        layer.bias = torch.tensor([[3.0, -3.0]])
        input = torch.tensor([[3.0]])
        out = layer(input)
        self.assertEqual(out.tolist(), [[4.0, -4.0]])

    def test_backward_pass_changes_weight_gradient(self) -> None:
        layer = QLinear(in_features=1, out_features=2, quantizer=Binarize())
        layer.weight = torch.tensor([[0.6], [0.7]])
        old_grad = layer.parametrizations.weight.original.grad
        layer.bias = torch.tensor([[3.0, -3.0]])
        input = torch.tensor([[3.0]])
        out = layer(input)
        loss = (out - torch.tensor([-1.0])).sum()
        loss.backward()
        new_weight_grad = layer.parametrizations.weight.original.grad
        self.assertNotEqual(new_weight_grad[0], old_grad)
        self.assertNotEqual(new_weight_grad[1], old_grad)

    def test_binarized_linear_layer_with_weight_clipper(self) -> None:
        layer = QLinear(
            in_features=1,
            out_features=2,
            quantizer=Binarize(),
            constraints=[WeightClipper()],
        )
        layer.weight = torch.tensor([[-2.0], [2.0]])
        layer.bias = torch.tensor([[2.0, 2.0]])
        input = torch.tensor([[3.0]])
        with torch.no_grad():
            layer.apply_constraint()
        out = layer(input)
        self.assertEqual(out.tolist(), [[-2, 4]])

    def test_throw_error_if_quantizer_is_none(self) -> None:
        with self.assertRaises(TypeError):
            _ = QLinear(in_features=1, out_features=1, quantizer=None)


class QConv1dTest(unittest.TestCase):
    def test_binarized_call_with_bias(self) -> None:
        layer = QConv1d(
            in_channels=1, out_channels=1, kernel_size=(2,), quantizer=Binarize()
        )
        layer.weight = torch.zeros(1, 1, 2)
        layer.bias = torch.tensor([3.0])
        input = torch.tensor([[[3.0, 2.0]]])
        out = layer(input)
        self.assertEqual(out.tolist(), [[[6.0]]])

    def test_backward_pass_changes_weight_gradient(self) -> None:
        layer = QConv1d(
            in_channels=1, out_channels=1, kernel_size=(2,), quantizer=Binarize()
        )
        layer.weight = torch.zeros(1, 1, 2)
        layer.bias = torch.tensor([3.0])
        old_grad = layer.parametrizations.weight.original.grad
        input = torch.tensor([[[3.0, 2.0]]])
        out = layer(input)
        loss = (out - torch.tensor([-1.0])).sum()
        loss.backward()
        new_weight_grad = layer.parametrizations.weight.original.grad
        self.assertNotEqual(new_weight_grad[0], old_grad)

    def test_call_with_weight_clipper_constraint(self) -> None:
        layer = QConv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            quantizer=Binarize(),
            constraints=[WeightClipper()],
        )
        layer.weight = torch.tensor([[[-2.0, 2.0]]])
        layer.bias = torch.tensor([3.0])
        input = torch.tensor([[[3.0, 2.0]]])
        with torch.no_grad():
            layer.apply_constraint()
        out = layer(input)
        self.assertEqual(out.tolist(), [[[0.0]]])

    def test_apply_constraint_if_constraint_is_none(self) -> None:
        layer = QConv1d(
            in_channels=1, out_channels=1, kernel_size=2, quantizer=Binarize()
        )
        layer.weight = torch.tensor([[[-2.0, 2.0]]])
        layer.bias = torch.tensor([3.0])
        input = torch.tensor([[[3.0, 2.0]]])
        with torch.no_grad():
            layer.apply_constraint()
        out = layer(input)
        self.assertEqual(out.tolist(), [[[0.0]]])

    def test_throw_error_if_quantizer_is_none(self) -> None:
        with self.assertRaises(TypeError):
            _ = QConv1d(in_channels=1, out_channels=1, kernel_size=2, quantizer=None)


class QConv2dTest(unittest.TestCase):
    def test_binarized_call_with_bias(self) -> None:
        layer = QConv2d(
            in_channels=1, out_channels=1, kernel_size=2, quantizer=Binarize()
        )
        layer.weight = torch.zeros(1, 1, 2, 2)
        layer.bias = torch.tensor([3.0])
        input = torch.tensor([[[[3.0, 2.0], [3.0, 2.0]]]])
        out = layer(input)
        self.assertEqual(out.tolist(), [[[[11.0]]]])

    def test_backward_pass_changes_weight_gradient(self) -> None:
        layer = QConv2d(
            in_channels=1, out_channels=1, kernel_size=2, quantizer=Binarize()
        )
        layer.weight = torch.zeros(1, 1, 2, 2)
        layer.bias = torch.tensor([3.0])
        input = torch.tensor([[[[3.0, 2.0], [3.0, 2.0]]]])
        old_grad = layer.parametrizations.weight.original.grad
        out = layer(input)
        loss = (out - torch.tensor([-1.0])).sum()
        loss.backward()
        new_weight_grad = layer.parametrizations.weight.original.grad
        self.assertNotEqual(new_weight_grad[0], old_grad)

    def test_call_with_weight_clipper_constraint(self) -> None:
        layer = QConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            quantizer=Binarize(),
            constraints=[WeightClipper()],
        )
        layer.weight = torch.tensor([[[[-2.0, 2.0], [0.0, 1.0]]]])
        layer.bias = torch.tensor([3.0])
        input = torch.tensor([[[[3.0, 2.0], [3.0, 2.0]]]])
        with torch.no_grad():
            layer.apply_constraint()
        out = layer(input)
        self.assertEqual(out.tolist(), [[[[5.0]]]])

    def test_throw_error_if_quantizer_is_none(self) -> None:
        with self.assertRaises(TypeError):
            _ = QConv2d(in_channels=1, out_channels=1, kernel_size=2, quantizer=None)


class LayerTests(TensorTestCase):
    def test_TernaryConvolution(self):
        layer = TernaryConvolution1d(
            in_channels=1, out_channels=2, kernel_size=2, groups=1, bias=False
        )
        self.assertEqual(type(layer.quantize), type(Ternarize()))
        self.assertEqual(layer.channel_multiplexing_factor, 1)

    def test_MultilevelResidualBinarizationConv(self):
        layer = MultilevelResidualBinarizationConv1d(
            in_channels=1, out_channels=2, kernel_size=2, groups=1, bias=False
        )
        self.assertEqual(type(layer.quantize), type(QuantizeTwoBit()))
        self.assertEqual(layer.channel_multiplexing_factor, 2)

    def test_BinaryActivatedConv(self):
        layer = BinaryActivatedConv1d(
            in_channels=1, out_channels=2, kernel_size=2, groups=1, bias=False
        )
        self.assertEqual(type(layer.quantize), type(Binarize()))
        self.assertEqual(layer.channel_multiplexing_factor, 1)

    def test_SplitConvolutionBase(self):
        layer_function = partial(
            BatchNormedActivatedConv1d,
            activation=Binarize,
            channel_multiplexing_factor=1,
        )
        layer = SplitConvolutionBase(
            in_channels=2,
            out_channels=4,
            kernel_size=2,
            convolution=layer_function,
            codomain_elements=[-1, 1],
        )
        layer.depthwise.conv.weight = torch.nn.Parameter(
            torch.ones_like(layer.depthwise.conv.weight)
        )
        layer.pointwise.conv.weight = torch.nn.Parameter(
            torch.ones_like(layer.pointwise.conv.weight)
        )
        test_input = torch.ones((2, 2, 3))
        output = layer(test_input)
        expected = torch.ones(2, 4, 2)
        self.assertTrue(torch.all((expected == output)))
        self.assertEqual(layer.depthwise.conv.groups, 2)

    def test_BinarySplitConv(self):
        layer = BinarySplitConv(in_channels=1, out_channels=2, kernel_size=2)
        self.assertEqual(type(layer.depthwise.quantize), type(Binarize()))
        self.assertEqual(layer.depthwise.channel_multiplexing_factor, 1)
        self.assertEqual(type(layer.pointwise.quantize), type(Binarize()))

    def test_TernarySplitConv(self):
        layer = TernarySplitConv(in_channels=1, out_channels=2, kernel_size=2)
        self.assertEqual(type(layer.depthwise.quantize), type(Ternarize()))
        self.assertEqual(layer.depthwise.channel_multiplexing_factor, 1)
        self.assertEqual(type(layer.pointwise.quantize), type(Ternarize()))

    def test_TwoBitSplitConv(self):
        layer = TwoBitSplitConv(in_channels=1, out_channels=2, kernel_size=2)
        self.assertEqual(type(layer.depthwise.quantize), type(QuantizeTwoBit()))
        self.assertEqual(layer.depthwise.channel_multiplexing_factor, 2)
        self.assertEqual(type(layer.pointwise.quantize), type(QuantizeTwoBit()))

    @SkipTest
    def test_QLSTMCell(self):
        with self.subTest("full res QLSTM cell equal PyTorch LSTM cell (with bias)"):
            lstm_cell = torch.nn.LSTMCell(input_size=3, hidden_size=5, bias=True)

            def identity(x):
                return x

            qlstm_cell = QLSTMCell(
                input_size=3,
                hidden_size=5,
                bias=True,
                weight_quantizer=identity,
                state_quantizer=lambda x: x,
            )
            qlstm_cell.weight_ih = lstm_cell.weight_ih
            qlstm_cell.weight_hh = lstm_cell.weight_hh
            qlstm_cell.bias_ih = lstm_cell.bias_ih
            qlstm_cell.bias_hh = lstm_cell.bias_hh
            inp = torch.rand(1, 3)
            lstm_h1, lstm_c1 = lstm_cell(inp)
            qlstm_h1, qlstm_c1 = qlstm_cell(inp)
            self.assertTrue(torch.equal(lstm_h1, qlstm_h1))
            self.assertTrue(torch.equal(lstm_c1, qlstm_c1))

        with self.subTest("full res QLSTM cell equal PyTorch LSTM cell (without bias)"):
            lstm_cell = torch.nn.LSTMCell(input_size=3, hidden_size=5, bias=False)
            qlstm_cell = QLSTMCell(
                input_size=3,
                hidden_size=5,
                bias=False,
            )
            qlstm_cell.weight_ih = lstm_cell.weight_ih
            qlstm_cell.weight_hh = lstm_cell.weight_hh
            qlstm_cell.bias_ih = lstm_cell.bias_ih
            qlstm_cell.bias_hh = lstm_cell.bias_hh
            inp = torch.rand(1, 3)
            lstm_h1, lstm_c1 = lstm_cell(inp)
            qlstm_h1, qlstm_c1 = qlstm_cell(inp)
            self.assertTrue(torch.equal(lstm_h1, qlstm_h1))
            self.assertTrue(torch.equal(lstm_c1, qlstm_c1))

        with self.subTest("test binarized cell states and weights"):
            cell = QLSTMCell(
                input_size=1,
                hidden_size=1,
                bias=True,
                state_quantizer=Binarize(),
                weight_quantizer=Binarize(),
            )
            cell.weight_ih = torch.nn.Parameter(torch.ones_like(cell.weight_ih))
            cell.weight_hh = torch.nn.Parameter(torch.ones_like(cell.weight_hh) * (-1))
            cell.bias_ih = torch.nn.Parameter(torch.ones_like(cell.bias_ih))
            cell.bias_hh = torch.nn.Parameter(torch.ones_like(cell.bias_hh) * (-1))
            inp = torch.as_tensor([[1.0]])
            actual_outputs = cell(inp)
            actual_outputs = tuple(
                map(
                    lambda output: [
                        [round(n, 4) for n in row] for row in output.tolist()
                    ],
                    actual_outputs,
                )
            )
            target_outputs = ([[0.2311]], [[0.5]])
            self.assertListEqual(actual_outputs[0], target_outputs[0])
            self.assertListEqual(actual_outputs[1], target_outputs[1])

    def test_QLSTM(self):
        lstm_cell_id = "elasticai.creator.qat.layers.QLSTMCell"
        qlstm_parameters = {
            "input_size": 0,
            "hidden_size": 1,
            "state_quantizer": lambda x: x,
            "weight_quantizer": lambda x: x,
            "bias": False,
        }
        with patch(lstm_cell_id, new=MockQLSTMCell) as _:
            with self.subTest("test QLSTM default"):
                layer = QLSTM(**qlstm_parameters)
                input = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
                state = [torch.tensor([1.0]), torch.tensor([1.0])]
                outputs, cell_state = layer(input, state)
                self.assertEqual(outputs.tolist(), [[1.0], [1.0]])
                self.assertEqual(cell_state[0].tolist(), [1.0])
                self.assertEqual(cell_state[1].tolist(), [1.0])

        with patch(lstm_cell_id, new=MockQLSTMCell) as _:
            with self.subTest("test QLSTM other input + state"):
                layer = QLSTM(**qlstm_parameters)
                input = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
                state = [torch.tensor([5.0]), torch.tensor([-1.0])]
                outputs, cell_state = layer(input, state)
                self.assertEqual(outputs.tolist(), [[5.0], [5.0], [5.0]])
                self.assertEqual(cell_state[0].tolist(), [5.0])
                self.assertEqual(cell_state[1].tolist(), [-1.0])

    def test_shuffle_block_input_and_output_same_with_1_group(self):
        layer = ChannelShuffle(groups=1)
        input = torch.rand((2, 2, 3, 2))
        output = layer(input)
        n = torch.eq(input, output)
        self.assertTrue(torch.all(torch.eq(input, output)).item())

    def test_shuffle_block_input_and_output_with_2_group(self):
        layer = ChannelShuffle(groups=2)
        input = torch.tensor(
            [
                [[1], [2], [3], [4]],
            ]
        )
        output = layer(input)
        expected = torch.tensor(
            [
                [[1], [3], [2], [4]],
            ]
        )
        self.assertTrue(torch.all(torch.eq(output, expected)).item())
