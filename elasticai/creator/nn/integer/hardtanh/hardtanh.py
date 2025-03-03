import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.hardtanh.design import HardTanh as HardTanhDesign
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class HardTanh(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits,
            observer=GlobalMinMaxObserver(),
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits,
            observer=GlobalMinMaxObserver(),
        ).to(device)

        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)
        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> HardTanhDesign:
        pass

    def precompute(self):
        self.quantized_one = self.inputs_QParams.quantize(torch.tensor(1.0))
        self.quantized_minus_one = self.inputs_QParams.quantize(torch.tensor(-1.0))
        self.precomputed = True

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_outputs = torch.where(
            q_inputs > self.quantized_one, self.quantized_one, q_inputs
        )

        q_outputs = torch.where(
            q_outputs < self.quantized_minus_one, self.quantized_minus_one, q_outputs
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

        inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = self.hardtanh(inputs)
        self.outputs_QParams = self.inputs_QParams
        return outputs
