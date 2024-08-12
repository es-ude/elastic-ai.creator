import logging

import torch
import torch.nn.functional as F
from torch import nn

from elasticai.creator.nn.integer.config import DEVICE
from elasticai.creator.nn.integer.quant_utils.FakeQuantize import FakeQuantize
from elasticai.creator.nn.integer.quant_utils.Observers import MinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.quant_utils.QuantizedTensorValidator import (
    QuantizedTensorValidator,
)
from elasticai.creator.nn.integer.quant_utils.SaveQuantData import save_quant_data
from elasticai.creator.nn.integer.relu.design import ReLU as ReLUDesign
from elasticai.creator.vhdl.design_creator import DesignCreator


class ReLU(DesignCreator, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.input_QParams = QParams(
            is_symmetric=False, quant_bits=self.quant_bits, observer=MinMaxObserver()
        ).to(DEVICE)

    def create_design(self, name: str) -> ReLUDesign:
        return ReLUDesign(
            name=name,
            data_width=self.quant_bits,
            threshold=int(self.input_QParams.zero_point.detach()),
            clock_option=False,
        )

    def int_forward(self, input: torch.IntTensor) -> torch.FloatTensor:
        # quantise input if input is of floating point type
        if input.dtype == torch.float32 or input.dtype == torch.float64:
            q_input = self.input_QParams.quantizeProcess(input)
        else:
            q_input = input
        QuantizedTensorValidator.check_dtype(
            q_input, "q_input", torch.int32, self.logger
        )
        QuantizedTensorValidator.check_drange(
            q_input,
            "q_input",
            -(2 ** (self.quant_bits - 1)),
            (2 ** (self.quant_bits - 1)) - 1,
            self.logger,
        )

        zero_point = self.input_QParams.zero_point
        q_input = q_input.to(DEVICE)
        q_input[q_input < zero_point] = zero_point

        QuantizedTensorValidator.check_dtype(
            q_input, "q_input-zero_point", torch.int32, self.logger
        )
        QuantizedTensorValidator.check_drange(
            q_input,
            "q_input-zero_point",
            zero_point,
            (2 ** (self.quant_bits - 1)) - 1,
            self.logger,
        )
        output = q_input

        return output

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        if self.training:
            self.input_QParams.updateScaleZeropoint(input)

        input = FakeQuantize.apply(input.to(DEVICE), self.input_QParams)

        output = F.relu(input)

        return output
