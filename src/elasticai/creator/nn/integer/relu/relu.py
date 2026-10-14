import torch
import torch.nn.functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.relu.design import ReLU as ReLUDesign
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class ReLU(DesignCreatorModule):
    """ReLU with quantization-aware training and integer inference.

    In the quantized domain, the float value 0 is represented by the zero
    point of the inputs, so the integer ReLU is `max(q_x, z_x)`. Inputs and
    outputs share their quantization parameters: after `forward`,
    `outputs_QParams` is `inputs_QParams`.
    """

    def __init__(
        self,
        *,
        name: str,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
        enable_error_analysis: bool = False,
    ):
        """Create the layer.

        Args:
            name: Name of the layer, used for the saved quantization data.
            quant_bits: Bit width of inputs and outputs.
            quant_data_dir: Directory for the quantized inputs and outputs
                saved by `int_forward`. `None` disables saving.
            device: Device of the quantization parameters.
            enable_error_analysis: Also save the float outputs of `forward`
                and the dequantized outputs of `int_forward`.
        """
        super().__init__()

        self.name = name
        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir
        self.enable_error_analysis = enable_error_analysis

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits,
            observer=GlobalMinMaxObserver(),
        ).to(device)

        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits,
            observer=GlobalMinMaxObserver(),
        ).to(device)

    def create_design(self, name: str) -> ReLUDesign:
        """Create the VHDL design, with the zero point of the inputs as threshold.

        Args:
            name: Name of the VHDL entity.
        """
        return ReLUDesign(
            name=name,
            data_width=self.quant_bits,
            threshold=int(self.inputs_QParams.zero_point.detach()),
            clock_option=False,
            work_library_name="work",
        )

    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor:
        """Run integer-only inference, as the hardware does.

        Computes `max(q_x, z_x)`. Inputs and outputs are saved to
        `quant_data_dir`.

        Args:
            q_inputs: Inputs quantized with `inputs_QParams`.

        Returns:
            The quantized outputs, quantized with the same parameters.
        """
        assert not self.training, "int_forward should be called in eval mode"
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")
        zero_point = self.inputs_QParams.zero_point.to(q_inputs.device)
        q_outputs = torch.maximum(q_inputs, zero_point.clone().detach())
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        if self.enable_error_analysis:
            save_quant_data(
                self.outputs_QParams.dequantize(q_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_outputs

    def forward(
        self,
        inputs: torch.Tensor,
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        """Apply ReLU to float inputs, optionally simulating quantization.

        Args:
            inputs: Float inputs of any shape.
            given_inputs_QParams: Quantization parameters of the inputs,
                usually the output parameters of the previous layer. If given
                in training mode, they replace `inputs_QParams` instead of
                being learned from `inputs`.
            enable_simquant: Simulate quantization of the inputs. If `False`,
                the layer acts like `torch.nn.functional.relu`.

        Returns:
            The float outputs, with the shape of `inputs`.
        """
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = F.relu(inputs)

        if enable_simquant:
            self.outputs_QParams = self.inputs_QParams
            if self.enable_error_analysis:
                save_quant_data(
                    outputs,
                    self.quant_data_dir,
                    f"{self.name}_y",
                )
        return outputs
