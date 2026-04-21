import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.hardsigmoid.design import (
    HardSigmoid as HardSigmoidDesign,
)
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class HardSigmoid(DesignCreatorModule, nn.Module):
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
        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> HardSigmoidDesign:
        return HardSigmoidDesign(
            name=name,
            data_width=self.quant_bits,
            quantized_three=int(self.quantized_three),
            quantized_minus_three=int(self.quantized_minus_three),
            quantized_one=int(self.quantized_one),
            quantized_zero=int(self.quantized_zero),
            tmp=self.tmp,
            work_library_name="work",
        )

    def _customized_hard_sigmoid(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        outputs = torch.where(inputs <= -3.0, torch.zeros_like(inputs), inputs)
        outputs = torch.where(inputs >= 3.0, torch.ones_like(outputs), outputs)
        outputs = torch.where(
            (inputs > -3.0) & (inputs < 3.0), outputs / 8.0 + 0.5, outputs
        )
        return outputs

    def precompute(self):
        self.quantized_three = self.inputs_QParams.quantize(torch.tensor(3.0))
        self.quantized_minus_three = self.inputs_QParams.quantize(torch.tensor(-3.0))

        self.quantized_one = self.inputs_QParams.quantize(torch.tensor(1.0))
        self.quantized_zero = self.inputs_QParams.quantize(torch.tensor(0.0))

        # self.out_quantized_one = self.outputs_QParams.quantize(torch.tensor(1.0))
        # self.out_quantized_zero = self.outputs_QParams.quantize(torch.tensor(0.0))

        self.tmp = int(
            self.inputs_QParams.quantize(torch.tensor(0.5))
            - self.inputs_QParams.zero_point // 8
        )
        self.precomputed = True

    def _div_trunc(self, x: torch.Tensor, y: int) -> torch.Tensor:
        return torch.where(x >= 0, x // y, -((-x) // y))

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_outputs = torch.where(
            q_inputs <= self.quantized_minus_three,
            torch.ones_like(q_inputs) * self.quantized_zero,
            q_inputs,
        )

        q_outputs = torch.where(
            q_inputs >= self.quantized_three,
            torch.ones_like(q_outputs) * self.quantized_one,
            q_outputs,
        )

        q_outputs = torch.where(
            (q_inputs > self.quantized_minus_three) & (q_inputs < self.quantized_three),
            self._div_trunc(q_inputs, 8) + self.tmp,
            q_outputs,
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")

        return q_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = self._customized_hard_sigmoid(inputs)

        if enable_simquant:
            self.outputs_QParams = self.inputs_QParams
            outputs = SimQuant.apply(outputs, self.outputs_QParams)

        return outputs
