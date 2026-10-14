import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.maxpooling.design import (
    MaxPooling1d as MaxPooling1dDesign,
)
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class MaxPooling1d(DesignCreatorModule, nn.Module):
    """Max pooling over the sequence, with integer inference.

    Inputs have the shape `(batch, channels, seq_len)`. The layer takes the
    maximum of non-overlapping windows of `kernel_size` values (the stride
    equals `kernel_size`). `in_features` is the sequence length and
    `in_num_dimensions` the number of channels; with these, the VHDL design
    matches `int_forward` (checked with GHDL).

    Taking the maximum does not change the scale, so the layer needs no
    `precompute`, and after `forward`, `outputs_QParams` is `inputs_QParams`.
    """

    def __init__(
        self,
        *,
        name: str,
        in_features: int,
        out_features: int,
        in_num_dimensions: int,
        out_num_dimensions: int,
        kernel_size: int,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        """Create the layer.

        Args:
            name: Name of the layer, used for the saved quantization data.
            in_features: Sequence length of the inputs.
            out_features: Sequence length of the outputs,
                `in_features // kernel_size`.
            in_num_dimensions: Number of channels of the inputs.
            out_num_dimensions: Number of channels of the outputs.
            kernel_size: Size and stride of the pooling windows.
            quant_bits: Bit width of inputs and outputs.
            quant_data_dir: Directory for the quantized inputs and outputs
                saved by `int_forward`. `None` disables saving.
            device: Device of the quantization parameters.
        """
        super().__init__()

        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.in_num_dimensions = in_num_dimensions
        self.out_num_dimensions = out_num_dimensions
        self.kernel_size = kernel_size

        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.scale_factor_Math_ops = MathOperations()

    def create_design(self, name: str) -> MaxPooling1dDesign:
        """Create the VHDL design.

        Args:
            name: Name of the VHDL entity.
        """
        return MaxPooling1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            out_features=self.out_features,
            in_num_dimensions=self.in_num_dimensions,
            out_num_dimensions=self.out_num_dimensions,
            kernel_size=self.kernel_size,
            work_library_name="work",
            resource_option="auto",
        )

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run integer-only inference, as the hardware does.

        Inputs and outputs are saved to `quant_data_dir`.

        Args:
            q_inputs: Inputs quantized with `inputs_QParams`, of shape
                `(batch, channels, seq_len)`.

        Returns:
            The maximum of each window, of shape
            `(batch, channels, seq_len // kernel_size)`, quantized with the
            same parameters as the inputs.
        """
        assert not self.training, "int_forward should be called in eval mode"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        batch_size, channels, seq_len = q_inputs.shape
        kernel_size = self.kernel_size
        stride = self.kernel_size

        output_length = (seq_len - kernel_size) // stride + 1

        q_outputs = torch.empty(
            (batch_size, channels, output_length),
            dtype=torch.int32,
            device=q_inputs.device,
        )

        for i in range(output_length):
            start = i * stride
            end = start + kernel_size
            q_outputs[:, :, i] = q_inputs[:, :, start:end].max(dim=2)[0]

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self,
        inputs: torch.Tensor,
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        """Apply max pooling to float inputs, optionally simulating quantization.

        Args:
            inputs: Float inputs of shape `(batch, channels, seq_len)`.
            given_inputs_QParams: Quantization parameters of the inputs,
                usually the output parameters of the previous layer. If given
                in training mode, they replace `inputs_QParams` instead of
                being learned from `inputs`.
            enable_simquant: Simulate quantization of the inputs.

        Returns:
            The float outputs, of shape `(batch, channels, seq_len // kernel_size)`.
        """
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = F.max_pool1d(
            inputs, kernel_size=self.kernel_size, stride=self.kernel_size
        )

        self.outputs_QParams = self.inputs_QParams
        return outputs
