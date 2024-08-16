import logging

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.nn.integer.config import DEVICE
from elasticai.creator.nn.integer.linear.design import Linear as LinearDesign
from elasticai.creator.nn.integer.math_operations.addition import add
from elasticai.creator.nn.integer.math_operations.matrixmultiplication import matmul
from elasticai.creator.nn.integer.math_operations.subtraction import subtract
from elasticai.creator.nn.integer.quant_utils.BitShifting import (
    scaling_m,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.FakeQuantize import FakeQuantize
from elasticai.creator.nn.integer.quant_utils.Observers import MinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.quant_utils.QuantizedTensorValidator import (
    QuantizedTensorValidator,
)
from elasticai.creator.nn.integer.quant_utils.SaveQuantData import save_quant_data
from elasticai.creator.vhdl.design_creator import DesignCreator


class Linear(DesignCreator, nn.Linear):
    def __init__(self, **kwargs):
        super().__init__(
            kwargs.get("in_features"), kwargs.get("out_features"), kwargs.get("bias")
        )
        self.name = kwargs.get("name")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.quant_bits = kwargs.get("quant_bits")

        # TODO: quantization scheme for each quantiztaion objects should be chosen by the user
        self.weight_QParams = QParams(
            is_symmetric=False,
            quant_bits=kwargs.get("quant_bits"),
            observer=MinMaxObserver(),
        ).to(DEVICE)
        self.bias_QParams = QParams(
            is_symmetric=True,
            quant_bits=kwargs.get("quant_bits"),
            observer=MinMaxObserver(),
        ).to(DEVICE)
        self.input_QParams = QParams(
            is_symmetric=False,
            quant_bits=kwargs.get("quant_bits"),
            observer=MinMaxObserver(),
        ).to(DEVICE)
        self.output_QParams = QParams(
            is_symmetric=False,
            quant_bits=kwargs.get("quant_bits"),
            observer=MinMaxObserver(),
        ).to(DEVICE)

    def create_design(self, name: str) -> LinearDesign:
        # BUG: to make sure about if the self.q_weight and self.q_bias are lists. (?! Done ?!)
        return LinearDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            out_features=self.out_features,
            weights=self.q_weight.tolist(),
            bias=self.q_bias.tolist(),
            scaler=self.scale_factor_M_q.item(),
            shift=self.scale_factor_M_q_shift.item(),
            z_w=self.weight_QParams.zero_point.item(),
            z_b=self.bias_QParams.zero_point.item(),
            z_x=self.input_QParams.zero_point.item(),
            z_y=self.output_QParams.zero_point.item(),
        )

    def _quant_weight(
        self, weight: torch.FloatTensor, weight_QParams: torch.nn.Module
    ) -> torch.IntTensor:
        q_weight = weight_QParams.quantizeProcess(weight)

        if not weight_QParams.is_symmetric:
            q_weight = subtract(
                q_weight, weight_QParams.zero_point, weight_QParams.quant_bits + 1
            )

        return q_weight

    def _quant_bias(
        self,
        bias: torch.FloatTensor,
        bias_QParams: torch.nn.Module,
        given_scale_factor: torch.FloatTensor,
        given_quant_bits: int,
    ) -> torch.IntTensor:
        bias_QParams.set_scale_factor(given_scale_factor)
        bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        bias_QParams.set_quant_range(given_quant_bits)

        q_bias = bias_QParams.quantizeProcess(bias)

        if not bias_QParams.is_symmetric:
            q_bias = subtract(q_bias, bias_QParams.zero_point, given_quant_bits + 1)

        return q_bias

    def precompute(self) -> None:
        self.q_weight = self._quant_weight(
            weight=self.weight, weight_QParams=self.weight_QParams
        )

        new_bias_scale_factor = self.input_QParams.scale * self.weight_QParams.scale
        new_bias_quant_bits = (self.input_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        self.tmp_quant_bits = new_bias_quant_bits
        self.q_bias = self._quant_bias(
            given_scale_factor=new_bias_scale_factor,
            bias_QParams=self.bias_QParams,
            bias=self.bias.to(DEVICE),
            given_quant_bits=new_bias_quant_bits,
        )

        self.scale_factor_M = (
            self.input_QParams.scale
            * self.weight_QParams.scale
            / self.output_QParams.scale
        )

        self.scale_factor_M_q_shift, self.scale_factor_M_q = scaling_m(
            self.scale_factor_M
        )

    def int_forward(
        self,
        q_input: torch.IntTensor,
    ) -> torch.IntTensor:
        q_input = subtract(
            q_input, self.input_QParams.zero_point, self.input_QParams.quant_bits + 1
        )

        tmp_quant_bits = (self.input_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        tmp = matmul(
            q_input,
            self.q_weight.t(),
            tmp_quant_bits,  # TODO further +1 or not
        )

        if self.bias is not None:
            tmp = add(tmp, self.q_bias, self.tmp_quant_bits + 1)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_M_q_shift, self.scale_factor_M_q
        ).to("cpu")
        QuantizedTensorValidator.check_dtype(
            tmp, "tmp after bit_shifting", torch.int32, self.logger
        )
        QuantizedTensorValidator.check_drange(
            tmp,
            "tmp after bit_shifting",
            -(2 ** (self.quant_bits + 1)),
            (2 ** (self.quant_bits + 1)) - 1,
            self.logger,
        )

        # process output
        output = add(tmp, self.output_QParams.zero_point, self.quant_bits)

        return output.to(DEVICE)

    def forward(
        self, input: torch.FloatTensor, given_input_QParams: QParams = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_input_QParams is not None:
                self.input_QParams = given_input_QParams
            else:
                self.input_QParams.updateScaleZeropoint(
                    input
                )  # required for FakeQuantize even if scaler_mode is fixed

        if self.training:
            # self.input_QParams.updateScaleZeropoint(input)
            self.weight_QParams.updateScaleZeropoint(self.weight)
            self.bias_QParams.updateScaleZeropoint(self.bias)

        input = FakeQuantize.apply(input, self.input_QParams)
        weight = FakeQuantize.apply(self.weight.to(DEVICE), self.weight_QParams)
        bias = FakeQuantize.apply(self.bias.to(DEVICE), self.bias_QParams)

        output = F.linear(input, weight, bias)

        if self.training:
            self.output_QParams.updateScaleZeropoint(output)

        output = FakeQuantize.apply(output, self.output_QParams)
        return output
