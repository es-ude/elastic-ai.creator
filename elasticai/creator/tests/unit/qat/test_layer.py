import unittest
from functools import partial
from unittest import SkipTest

import torch

from elasticai.creator.qat.constraints import WeightClipper
from elasticai.creator.qat.layers import (
    QLSTM,
    BatchNormedActivatedConv1d,
    Binarize,
    BinaryConv1d,
    BinarySplitConv1d,
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
    TernaryConv1d,
    TernarySplitConv1d,
    TwoBitSplitConv1d,
)


def round_tensor(
    *input_tensor: torch.Tensor, decimals: int
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.to(float).detach().apply_(lambda x: round(x, decimals))
        for tensor in input_tensor
    )


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


class BatchNormedActivatedConv1dTest(unittest.TestCase):
    def test_binarized_call_without_bias(self) -> None:
        layer = BatchNormedActivatedConv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=2,
            groups=1,
            bias=False,
            activation=Binarize,
            channel_multiplexing_factor=1,
        )
        layer.conv.weight = torch.nn.Parameter(torch.ones_like(layer.conv.weight))
        test_input = torch.ones((2, 1, 3))
        output = layer(test_input)
        expected = torch.ones(2, 2, 2)
        self.assertTrue(torch.all((expected == output)))


class DefineBatchNormedConvolution1dTest(unittest.TestCase):
    def test_binary_conv1d(self) -> None:
        layer = BinaryConv1d(
            in_channels=1, out_channels=2, kernel_size=2, groups=1, bias=False
        )
        self.assertEqual(type(layer.quantize), type(Binarize()))
        self.assertEqual(layer.channel_multiplexing_factor, 1)

    def test_ternary_conv1d(self) -> None:
        layer = TernaryConv1d(
            in_channels=1, out_channels=2, kernel_size=2, groups=1, bias=False
        )
        self.assertEqual(type(layer.quantize), type(Ternarize()))
        self.assertEqual(layer.channel_multiplexing_factor, 1)

    def test_multilevel_residual_binarization_conv1d(self) -> None:
        layer = MultilevelResidualBinarizationConv1d(
            in_channels=1, out_channels=2, kernel_size=2, groups=1, bias=False
        )
        self.assertEqual(type(layer.quantize), type(QuantizeTwoBit()))
        self.assertEqual(layer.channel_multiplexing_factor, 2)


class SplitConvolutionBaseTest(unittest.TestCase):
    def test_binarized_conv1d_outputs_as_expected(self) -> None:
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
        self.assertTrue(torch.all(expected == output))
        self.assertEqual(layer.depthwise.conv.groups, 2)


class DefineSplitConvolutionTest(unittest.TestCase):
    def test_binary_split_conv1d(self) -> None:
        layer = BinarySplitConv1d(in_channels=1, out_channels=2, kernel_size=2)
        self.assertEqual(type(layer.depthwise.quantize), type(Binarize()))
        self.assertEqual(layer.depthwise.channel_multiplexing_factor, 1)
        self.assertEqual(type(layer.pointwise.quantize), type(Binarize()))

    def test_ternary_split_conv1d(self) -> None:
        layer = TernarySplitConv1d(in_channels=1, out_channels=2, kernel_size=2)
        self.assertEqual(type(layer.depthwise.quantize), type(Ternarize()))
        self.assertEqual(layer.depthwise.channel_multiplexing_factor, 1)
        self.assertEqual(type(layer.pointwise.quantize), type(Ternarize()))

    def test_two_bit_split_conv1d(self) -> None:
        layer = TwoBitSplitConv1d(in_channels=1, out_channels=2, kernel_size=2)
        self.assertEqual(type(layer.depthwise.quantize), type(QuantizeTwoBit()))
        self.assertEqual(layer.depthwise.channel_multiplexing_factor, 2)
        self.assertEqual(type(layer.pointwise.quantize), type(QuantizeTwoBit()))


class ChannelShuffleTest(unittest.TestCase):
    def test_input_and_output_same_with_1_group(self) -> None:
        layer = ChannelShuffle(groups=1)
        input = torch.rand((2, 2, 3, 2))
        output = layer(input)
        self.assertEqual(input.tolist(), output.tolist())

    def test_input_and_output_with_2_groups(self) -> None:
        layer = ChannelShuffle(groups=2)
        input = torch.tensor([[[1], [2], [3], [4]]])
        output = layer(input)
        expected = torch.tensor([[[1], [3], [2], [4]]])
        self.assertEqual(output.tolist(), expected.tolist())


class QLSTMCellTest(unittest.TestCase):
    def test_full_res_qlstm_cell_equal_pytorch_lstm_cell_without_bias(self) -> None:
        lstm_cell = torch.nn.LSTMCell(input_size=3, hidden_size=5, bias=False)
        qlstm_cell = QLSTMCell(input_size=3, hidden_size=5, bias=False)

        qlstm_cell.weight_ih = lstm_cell.weight_ih
        qlstm_cell.weight_hh = lstm_cell.weight_hh
        qlstm_cell.bias_ih = lstm_cell.bias_ih
        qlstm_cell.bias_hh = lstm_cell.bias_hh

        inp = torch.rand(1, 3)
        lstm_h1, lstm_c1 = lstm_cell(inp)
        qlstm_h1, qlstm_c1 = qlstm_cell(inp)

        lstm_h1, lstm_c1 = round_tensor(lstm_h1, lstm_c1, decimals=4)
        qlstm_h1, qlstm_c1 = round_tensor(qlstm_h1, qlstm_c1, decimals=4)

        self.assertEqual(lstm_h1.tolist(), qlstm_h1.tolist())
        self.assertEqual(lstm_c1.tolist(), qlstm_c1.tolist())

    def test_full_res_qlstm_cell_equal_pytorch_lstm_cell_with_bias(self) -> None:
        lstm_cell = torch.nn.LSTMCell(input_size=3, hidden_size=5, bias=True)
        qlstm_cell = QLSTMCell(input_size=3, hidden_size=5, bias=True)

        qlstm_cell.weight_ih = lstm_cell.weight_ih
        qlstm_cell.weight_hh = lstm_cell.weight_hh
        qlstm_cell.bias_ih = lstm_cell.bias_ih
        qlstm_cell.bias_hh = lstm_cell.bias_hh

        inp = torch.rand(1, 3)
        lstm_h1, lstm_c1 = lstm_cell(inp)
        qlstm_h1, qlstm_c1 = qlstm_cell(inp)

        lstm_h1, lstm_c1 = round_tensor(lstm_h1, lstm_c1, decimals=4)
        qlstm_h1, qlstm_c1 = round_tensor(qlstm_h1, qlstm_c1, decimals=4)

        self.assertEqual(lstm_h1.tolist(), qlstm_h1.tolist())
        self.assertEqual(lstm_c1.tolist(), qlstm_c1.tolist())

    def test_binarized_state_and_weight(self) -> None:
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
        actual_h1, actual_c1 = cell(inp)
        actual_h1, actual_c1 = round_tensor(actual_h1, actual_c1, decimals=4)
        target_h1, target_c1 = [[0.2311]], [[0.5]]

        self.assertEqual(target_h1, actual_h1.tolist())
        self.assertEqual(target_c1, actual_c1.tolist())

    def test_output_shape_on_flat_input_data(self) -> None:
        input_size = 2
        hidden_size = 4
        cell = QLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)
        inputs = torch.rand(input_size)
        h1, c1 = cell(inputs)
        self.assertEqual(h1.shape, (hidden_size,))
        self.assertEqual(c1.shape, (hidden_size,))

    def test_output_shape_on_batched_input_data(self) -> None:
        input_size = 2
        hidden_size = 4
        batch_size = 3
        cell = QLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)
        inputs = torch.rand(batch_size, input_size)
        h1, c1 = cell(inputs)
        self.assertEqual(h1.shape, (batch_size, hidden_size))
        self.assertEqual(c1.shape, (batch_size, hidden_size))


class QLSTMTest(unittest.TestCase):
    def test_output_shape_on_flat_input_data(self) -> None:
        input_size, hidden_size, sequence_len = 2, 4, 3
        lstm = QLSTM(input_size, hidden_size, bias=True)
        inputs = torch.rand(sequence_len, input_size)
        h0, c0 = torch.rand(1, hidden_size), torch.rand(1, hidden_size)
        output, (h1, c1) = lstm(inputs, (h0, c0))
        self.assertEqual(output.shape, (sequence_len, hidden_size))
        self.assertEqual(h1.shape, (1, hidden_size))
        self.assertEqual(c1.shape, (1, hidden_size))

    def test_output_shape_on_batched_input_data(self) -> None:
        input_size, hidden_size, sequence_len = 2, 4, 3
        batch_size = 5
        lstm = QLSTM(input_size, hidden_size, bias=True)
        inputs = torch.rand(sequence_len, batch_size, input_size)
        h0 = torch.rand(1, batch_size, hidden_size)
        c0 = torch.rand(1, batch_size, hidden_size)
        output, (h1, c1) = lstm(inputs, (h0, c0))
        self.assertEqual(output.shape, (sequence_len, batch_size, hidden_size))
        self.assertEqual(h1.shape, (1, batch_size, hidden_size))
        self.assertEqual(c1.shape, (1, batch_size, hidden_size))

    def test_output_shape_on_batched_input_data_batch_first(self) -> None:
        input_size, hidden_size, sequence_len = 2, 4, 3
        batch_size = 5
        lstm = QLSTM(input_size, hidden_size, bias=True, batch_first=True)
        inputs = torch.rand(batch_size, sequence_len, input_size)
        h0 = torch.rand(1, batch_size, hidden_size)
        c0 = torch.rand(1, batch_size, hidden_size)
        output, (h1, c1) = lstm(inputs, (h0, c0))
        self.assertEqual(output.shape, (batch_size, sequence_len, hidden_size))
        self.assertEqual(h1.shape, (1, batch_size, hidden_size))
        self.assertEqual(c1.shape, (1, batch_size, hidden_size))

    def test_forward_without_explicit_given_state(self) -> None:
        pt_lstm = torch.nn.LSTM(input_size=1, hidden_size=2, bias=True)
        qlstm = QLSTM(input_size=1, hidden_size=2, bias=True)

        qlstm.cell.weight_ih = pt_lstm.weight_ih_l0
        qlstm.cell.weight_hh = pt_lstm.weight_hh_l0
        qlstm.cell.bias_ih = pt_lstm.bias_ih_l0
        qlstm.cell.bias_hh = pt_lstm.bias_hh_l0

        inputs = torch.tensor([[1.0], [1.0]])
        pt_outputs, (pt_h, pt_c) = pt_lstm(inputs)
        q_outputs, (q_h, q_c) = qlstm(inputs)

        pt_outputs, pt_h, pt_c = round_tensor(pt_outputs, pt_h, pt_c, decimals=4)
        q_outputs, q_h, q_c = round_tensor(q_outputs, q_h, q_c, decimals=4)

        self.assertEqual(pt_outputs.tolist(), q_outputs.tolist())
        self.assertEqual(pt_h.tolist(), q_h.tolist())
        self.assertEqual(pt_c.tolist(), q_c.tolist())

    def test_forward_with_input_and_state(self) -> None:
        pt_lstm = torch.nn.LSTM(input_size=1, hidden_size=2, bias=True)
        qlstm = QLSTM(input_size=1, hidden_size=2, bias=True)

        qlstm.cell.weight_ih = pt_lstm.weight_ih_l0
        qlstm.cell.weight_hh = pt_lstm.weight_hh_l0
        qlstm.cell.bias_ih = pt_lstm.bias_ih_l0
        qlstm.cell.bias_hh = pt_lstm.bias_hh_l0

        inputs = torch.tensor([[1.0], [1.0]])
        state = (torch.tensor([[5.0, 5.0]]), torch.tensor([[-1.0, -1.0]]))
        pt_outputs, (pt_h, pt_c) = pt_lstm(inputs, state)
        q_outputs, (q_h, q_c) = qlstm(inputs, state)

        pt_outputs, pt_h, pt_c = round_tensor(pt_outputs, pt_h, pt_c, decimals=4)
        q_outputs, q_h, q_c = round_tensor(q_outputs, q_h, q_c, decimals=4)

        self.assertEqual(pt_outputs.tolist(), q_outputs.tolist())
        self.assertEqual(pt_h.tolist(), q_h.tolist())
        self.assertEqual(pt_c.tolist(), q_c.tolist())
