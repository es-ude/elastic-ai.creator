import logging

import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.hardtanh.design import HardTanh as HardTanhDesign
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant


class HardTanh(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.device = kwargs.get("device")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits,
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits,
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)

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

        q_outputs = torch.where(
            q_inputs > self.quantized_one, self.quantized_one, q_inputs
        )
        q_outputs = torch.where(
            q_outputs < self.quantized_minus_one, self.quantized_minus_one, q_outputs
        )
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

        outputs = self.hardtanh(inputs)

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)

        outputs = SimQuant.apply(outputs, self.outputs_QParams)

        return outputs
