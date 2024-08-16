import logging

import torch
import torch.nn.functional as F
from torch import nn

from elasticai.creator.file_generation.savable import Path
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

        self.output_QParams = QParams(
            is_symmetric=False, quant_bits=self.quant_bits, observer=MinMaxObserver()
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
        q_output = q_input
        return q_output

    def forward(
        self, input: torch.FloatTensor, given_input_QParams: QParams = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_input_QParams is not None:
                self.input_QParams = given_input_QParams
            else:
                self.input_QParams.updateScaleZeropoint(input)

        input = FakeQuantize.apply(input.to(DEVICE), self.input_QParams)

        output = F.relu(input)

        self.output_QParams = self.input_QParams
        output = FakeQuantize.apply(output, self.output_QParams)

        return output
