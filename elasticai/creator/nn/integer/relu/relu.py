import logging

import torch
import torch.nn.functional as F
from torch import nn

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

        self.input_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"),
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

        self.output_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"),
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def create_design(self, name: str) -> ReLUDesign:
        return ReLUDesign(
            name=name,
            data_width=self.quant_bits,
            threshold=int(self.input_QParams.zero_point.detach()),
            clock_option=False,
            work_library_name="work",
        )

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        zero_point = self.input_QParams.zero_point.to(q_inputs.device)
        q_outputs = torch.maximum(q_inputs, zero_point.clone().detach())
        return q_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_input_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_input_QParams is None:
                self.input_QParams.update_quant_params(inputs)
            else:
                self.input_QParams = given_input_QParams

        inputs = SimQuant.apply(inputs, self.input_QParams)

        outputs = F.relu(inputs)

        # TODO: check if do fake quantization for output or not
        self.output_QParams = self.input_QParams
        outputs = SimQuant.apply(outputs, self.output_QParams)

        return outputs
