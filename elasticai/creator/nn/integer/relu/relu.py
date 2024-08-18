import logging

import torch
import torch.nn.functional as F
from torch import nn

from elasticai.creator.nn.integer.config import DEVICE
from elasticai.creator.nn.integer.quant_utils.Observers import MinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.relu.design import ReLU as ReLUDesign
from elasticai.creator.vhdl.design_creator import DesignCreator


class ReLU(DesignCreator, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.input_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"),
            observer=MinMaxObserver(),
        ).to(DEVICE)

        self.output_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"),
            observer=MinMaxObserver(),
        ).to(DEVICE)

    def create_design(self, name: str) -> ReLUDesign:
        return ReLUDesign(
            name=name,
            data_width=self.quant_bits,
            threshold=int(self.input_QParams.zero_point.detach()),
            clock_option=False,
        )

    def int_forward(self, q_input: torch.IntTensor) -> torch.FloatTensor:
        zero_point = self.input_QParams.zero_point
        q_input = q_input.to(DEVICE)
        q_output = torch.maximum(q_input, zero_point.clone().detach())
        return q_output

    def forward(
        self, input: torch.FloatTensor, given_input_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_input_QParams is None:
                self.input_QParams.update_quant_params(input)
            else:
                self.input_QParams = given_input_QParams

        input = SimQuant.apply(input.to(DEVICE), self.input_QParams)

        output = F.relu(input)

        # TODO: check if do fake quantization for output or not
        self.output_QParams = self.input_QParams
        output = SimQuant.apply(output, self.output_QParams)

        return output
