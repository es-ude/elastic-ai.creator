import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.avgpooling1d.design import (
    AVGPooling1d as AVGPooling1dDesign,
)
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class AVGPooling1d(DesignCreatorModule, nn.Module):
    """Global average pooling over the sequence, with integer inference.

    Inputs have the shape `(batch, seq_len, channels)`. The layer averages
    over `seq_len` and returns `(batch, 1, channels)`.

    Note:
        `precompute` divides by `in_features`, so the integer path is only
        correct if `in_features` is the sequence length. The VHDL template
        instead steps through the inputs with a stride of `in_features`, and
        its outputs currently differ from `int_forward` (checked with GHDL).
    """

    def __init__(
        self,
        *,
        name: str,
        in_features: int,
        out_features: int,
        in_num_dimensions: int,
        out_num_dimensions: int,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        """Create the layer.

        Args:
            name: Name of the layer, used for the saved quantization data.
            in_features: Sequence length of the inputs; `precompute` divides
                the sum by it.
            out_features: Passed to the VHDL design.
            in_num_dimensions: Passed to the VHDL design.
            out_num_dimensions: Passed to the VHDL design.
            quant_bits: Bit width of inputs and outputs.
            quant_data_dir: Directory for the quantized inputs and outputs
                saved by `int_forward`. `None` disables saving.
            device: Device of the quantization parameters.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.in_num_dimensions = in_num_dimensions
        self.out_num_dimensions = out_num_dimensions

        self.name = name
        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> AVGPooling1dDesign:
        """Create the VHDL design. Call `precompute` first.

        Args:
            name: Name of the VHDL entity.
        """
        return AVGPooling1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            out_features=self.out_features,
            in_num_dimensions=self.in_num_dimensions,
            out_num_dimensions=self.out_num_dimensions,
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def precompute(self) -> None:
        """Compute the integer scale for `int_forward` and `create_design`.

        Approximates `M = s_x / (s_y * in_features)` by
        `m_q * 2**-m_q_shift`. Must be called in eval mode, after training.
        """
        assert not self.training, "int_forward should be called in eval mode"
        L = self.in_features  # e.g., seq_len
        self.scale_factor_M = torch.tensor(
            self.inputs_QParams.scale_factor.item()
            / (self.outputs_QParams.scale_factor.item() * L),
        )
        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M.clone().detach()
        )

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Run integer-only inference.

        Sums `q_x - z_x` over the sequence, scales the sum by `M` and adds
        `z_y`. Inputs and outputs are saved to `quant_data_dir`.

        Args:
            q_inputs: Inputs quantized with `inputs_QParams`, of shape
                `(batch, seq_len, channels)`.

        Returns:
            The quantized outputs, of shape `(batch, 1, channels)`.
        """
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        # assume that (batch_size, seq_len, channels) is the shape of q_inputs
        tmp = torch.sum(q_inputs, dim=1, keepdim=True).to(
            torch.int32
        )  # sum over seq_len

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
        """Average float inputs over the sequence, optionally simulating quantization.

        Args:
            inputs: Float inputs of shape `(batch, seq_len, channels)`.
            given_inputs_QParams: Quantization parameters of the inputs,
                usually the output parameters of the previous layer. If given
                in training mode, they replace `inputs_QParams` instead of
                being learned from `inputs`.
            enable_simquant: Simulate quantization of inputs and outputs.

        Returns:
            The float outputs, of shape `(batch, 1, channels)`.
        """
        # assume that (batch_size, seq_len, channels) is the shape of inputs
        inputs = inputs.permute(0, 2, 1)

        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams
            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        # assume that (batch_size, channels, seq_len) is the shape of inputs
        outputs = F.avg_pool1d(inputs, kernel_size=inputs.size(2))

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)

            outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs.permute(0, 2, 1)
