import copy

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.conv1d.design import Conv1d as Conv1dDesign
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import (
    get_padded_q_inputs,
    get_padding_len,
)
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Conv1d(DesignCreatorModule, nn.Conv1d):
    """1D convolution with quantization-aware training and integer inference.

    Inputs have the shape `(batch, in_channels, seq_len)` and outputs
    `(batch, out_channels, seq_len + 2 * padding - kernel_size + 1)`. Padded
    values are 0, which `int_forward` represents by the zero point of the
    inputs.

    The layer has three modes:

    - `forward` with `enable_simquant=True` simulates quantization of inputs,
      weights, bias and outputs. In training mode it also updates their
      quantization parameters.
    - `int_forward` runs integer-only inference on quantized inputs. Call
      `precompute` first.
    - `create_design` creates the VHDL design. Call `precompute` first.
    """

    def __init__(
        self,
        *,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str,
        seq_len: int,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        """Create the layer.

        Args:
            name: Name of the layer, used for the saved quantization data.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            padding: Number of zeros added on each side, or `"same"` for
                outputs as long as the inputs.
            seq_len: Sequence length of the inputs.
            quant_bits: Bit width of inputs, outputs and weights.
            quant_data_dir: Directory for the quantized inputs and outputs
                saved by `int_forward`. `None` disables saving.
            device: Device of the quantization parameters.
        """
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.padding_len = get_padding_len(self.padding, self.kernel_size)
        self.seq_len = seq_len

        self.name = name
        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> Conv1dDesign:
        """Create the VHDL design from the precomputed integer parameters.

        Call `precompute` first.

        Args:
            name: Name of the VHDL entity.
        """
        return Conv1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size[0],
            seq_len=self.seq_len,
            padding=self.padding,
            padding_len=self.padding_len,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.weight_QParams.zero_point.item(),
            z_b=self.q_bias_zero_point,
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(self) -> torch.Tensor:
        """Quantize the weights and subtract their zero point."""
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = self.weight_QParams.quantize(self.weight).to("cpu")

        if not self.weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                self.weight_QParams.zero_point,
                self.weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(self) -> tuple[torch.Tensor, QParams]:
        """Quantize the bias with the scale of the accumulator, `s_x * s_w`.

        The bias is quantized with a copy of `bias_QParams`, so that
        `bias_QParams` keeps the parameters learned in training and `forward`
        gives the same outputs before and after `precompute`.

        Returns:
            The quantized bias and the QParams it was quantized with.
        """
        assert not self.training, "int_forward should be called in eval mode"
        q_bias_QParams = copy.deepcopy(self.bias_QParams)
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        q_bias_QParams.set_scale_factor(new_bias_scale_factor)
        q_bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        q_bias_QParams.set_quant_range(self.tmp_quant_bits)
        q_bias = q_bias_QParams.quantize(self.bias).to("cpu")
        if not q_bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, q_bias_QParams.zero_point, self.tmp_quant_bits + 1
            )
        return q_bias, q_bias_QParams

    def precompute(self) -> None:
        """Compute the integer parameters for `int_forward` and `create_design`.

        Quantizes weights and bias and approximates the requantization scale
        `M = s_x * s_w / s_y` by `m_q * 2**-m_q_shift`. Must be called in eval
        mode, after training.
        """
        assert not self.training, "int_forward should be called in eval mode"
        self.q_weights = self._get_quantized_weights()
        self.tmp_quant_bits = (
            self.inputs_QParams.quant_bits + 1 + self.weight_QParams.quant_bits + 1
        )
        if self.bias is not None:
            self.q_bias, q_bias_QParams = self._get_quantized_bias()
            # Only the zero point is kept. Storing the QParams module would add
            # its buffers to the state_dict of the layer.
            self.q_bias_zero_point = int(q_bias_QParams.zero_point.item())
            self.tmp_quant_bits += 1

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run integer-only inference.

        Pads with `z_x`, subtracts `z_x`, convolves with the quantized weights
        and bias, scales the result by `M` and adds `z_y`. Inputs and outputs
        are saved to `quant_data_dir`.

        Args:
            q_inputs: Inputs quantized with `inputs_QParams`, of shape
                `(batch, in_channels, seq_len)`.

        Returns:
            The quantized outputs.
        """
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        # Pad with the zero point before subtracting it, so that the padded
        # values represent 0.
        q_inputs = get_padded_q_inputs(
            padding_len=self.padding_len,
            q_inputs=q_inputs,
            inputs_QParams=self.inputs_QParams,
        )
        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        if self.q_bias is not None:
            tmp = F.conv1d(
                q_inputs,
                self.q_weights,
                self.q_bias,
                padding=0,
            )
            tmp = self.math_ops.clamp_result(tmp, self.tmp_quant_bits + 1)
        else:
            tmp = F.conv1d(
                q_inputs,
                self.q_weights,
                padding=0,
            )
            tmp = self.math_ops.clamp_result(tmp, self.tmp_quant_bits)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self,
        inputs: torch.Tensor,
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        """Apply the convolution to float inputs, optionally simulating quantization.

        Args:
            inputs: Float inputs of shape `(batch, in_channels, seq_len)`.
            given_inputs_QParams: Quantization parameters of the inputs,
                usually the output parameters of the previous layer. If given
                in training mode, they replace `inputs_QParams` instead of
                being learned from `inputs`.
            enable_simquant: Simulate quantization of inputs, weights, bias
                and outputs. If `False`, the layer acts like `nn.Conv1d`.

        Returns:
            The float outputs.
        """
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

                self.weight_QParams.update_quant_params(self.weight)
                if self.bias is not None:
                    self.bias_QParams.update_quant_params(self.bias)

            inputs = SimQuant.apply(inputs, self.inputs_QParams)
            weight = SimQuant.apply(self.weight, self.weight_QParams)
            if self.bias is not None:
                bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        if self.bias is not None:
            outputs = F.conv1d(inputs, weight, bias, padding=self.padding)
        else:
            outputs = F.conv1d(inputs, weight, padding=self.padding)

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)
            outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
