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
    def __init__(
        self,
        *,
        name: str,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
        enable_error_analysis: bool = False,
    ):
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
        return ReLUDesign(
            name=name,
            data_width=self.quant_bits,
            threshold=int(self.inputs_QParams.zero_point.detach()),
            clock_option=False,
            work_library_name="work",
        )

    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor:
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
