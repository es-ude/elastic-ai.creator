import logging

import torch
import torch.nn.functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.relu.design import ReLU as ReLUDesign


class ReLU(DesignCreatorModule):
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

    def create_design(self, name: str) -> ReLUDesign:
        return ReLUDesign(
            name=name,
            data_width=self.quant_bits,
            threshold=int(self.inputs_QParams.zero_point.detach()),
            clock_option=False,
            work_library_name="work",
        )

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        zero_point = self.inputs_QParams.zero_point.to(q_inputs.device)
        q_outputs = torch.maximum(q_inputs, zero_point.clone().detach())
        return q_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

        inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = F.relu(inputs)

        self.outputs_QParams = self.inputs_QParams

        outputs = SimQuant.apply(outputs, self.outputs_QParams)

        return outputs
