import logging

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.nn.integer.config import DEVICE
from elasticai.creator.nn.integer.linear.design import Linear as LinearDesign
from elasticai.creator.nn.integer.math_operations.addition import add
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
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

        # TODO: quantization scheme for each quantiztaion objects should be chosen by the user
        self.weight_QParams = QParams(
            is_symmetric=False, quant_bits=self.quant_bits, observer=MinMaxObserver()
        ).to(DEVICE)
        self.bias_QParams = QParams(
            is_symmetric=True,  # Only Bias applied signed symmetric quantization
            quant_bits=self.quant_bits,
            observer=MinMaxObserver(),
        ).to(DEVICE)
        self.input_QParams = QParams(
            is_symmetric=False, quant_bits=self.quant_bits, observer=MinMaxObserver()
        ).to(DEVICE)
        self.output_QParams = QParams(
            is_symmetric=False, quant_bits=self.quant_bits, observer=MinMaxObserver()
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
            scaler=self.m_int.item(),
            shift=self.m_N_shifts.item(),
            z_w=self.weight_QParams.zero_point.item(),
            z_b=self.bias_QParams.zero_point.item(),
            z_x=self.input_QParams.zero_point.item(),
            z_y=self.output_QParams.zero_point.item(),
        )

    def _quant_weight(
        self, weight: torch.FloatTensor, weight_QParams: torch.nn.Module
    ) -> torch.FloatTensor:
        q_weight = weight_QParams.quantizeProcess(weight)
        lower_bound = -(2 ** (self.quant_bits - 1))
        upper_bound = (2 ** (self.quant_bits - 1)) - 1

        QuantizedTensorValidator.check_dtype(
            q_weight, "q_weight", torch.int32, self.logger
        )
        QuantizedTensorValidator.check_drange(
            q_weight, "q_weight", lower_bound, upper_bound, self.logger
        )

        if weight_QParams.is_symmetric == False:
            q_weight = subtract(
                q_weight, weight_QParams.zero_point, self.quant_bits + 1
            )

        return q_weight

    def _quant_bias(
        self,
        bias: torch.FloatTensor,
        bias_QParams: torch.nn.Module,
        given_quant_scale: torch.FloatTensor,
        given_quant_bits: int,
    ) -> torch.FloatTensor:
        bias_QParams.set_scale(given_quant_scale)
        bias_QParams.set_zero_point(torch.zeros((1)))
        min_quant = -(2 ** (given_quant_bits - 1)) + 1
        max_quant = (2 ** (given_quant_bits - 1)) - 1
        bias_QParams.set_quant_range(
            given_min_quant=min_quant, given_max_quant=max_quant
        )

        q_bias = bias_QParams.quantizeProcess(bias)

        QuantizedTensorValidator.check_dtype(q_bias, "q_bias", torch.int32, self.logger)
        QuantizedTensorValidator.check_drange(
            q_bias, "q_bias", min_quant, max_quant, self.logger
        )

        return q_bias

    def precompute(self) -> None:
        self.q_weight = self._quant_weight(
            weight=self.weight.to(DEVICE), weight_QParams=self.weight_QParams
        )

        new_bias_QParms_scale = self.input_QParams.scale * self.weight_QParams.scale
        self.tmp_quant_bits = 2 * (self.quant_bits + 1)
        self.q_bias = self._quant_bias(
            given_quant_scale=new_bias_QParms_scale,
            bias_QParams=self.bias_QParams,
            bias=self.bias.to(DEVICE),
            given_quant_bits=self.tmp_quant_bits,
        )

        self.m = new_bias_QParms_scale / self.output_QParams.scale
        QuantizedTensorValidator.check_dtype(self.m, "m", torch.float32, self.logger)

        self.m_N_shifts, self.m_int = scaling_m(self.m)
        QuantizedTensorValidator.check_dtype(
            self.m_N_shifts, "m_N_shifts", torch.int32, self.logger
        )
        QuantizedTensorValidator.check_dtype(
            self.m_int, "m_int", torch.int32, self.logger
        )

    def int_forward(
        self,
        input: torch.Tensor,
        quant_data_file_dir: Path = None,
    ) -> torch.IntTensor:
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

        if quant_data_file_dir is not None:
            save_quant_data(q_input, quant_data_file_dir, f"{self.name}_q_x")

        q_input = subtract(q_input, self.input_QParams.zero_point, self.quant_bits + 1)

        # integer-only matrix multiplication on CPU
        q_input = q_input.to("cpu")
        tmp = q_input.matmul(self.q_weight.t().to("cpu"))
        QuantizedTensorValidator.check_dtype(tmp, "tmp", torch.int32, self.logger)
        QuantizedTensorValidator.check_drange(
            tmp,
            "integer-only matmul",
            -(2 ** (self.tmp_quant_bits)),
            (2 ** (self.tmp_quant_bits)) - 1,
            self.logger,
        )

        if self.bias is not None:
            tmp = add(tmp, self.q_bias.to("cpu"), self.tmp_quant_bits + 1)

        tmp = simulate_bitshifting(tmp, self.m_N_shifts, self.m_int).to("cpu")
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
        output = add(tmp, self.output_QParams.zero_point.to("cpu"), self.quant_bits)

        if quant_data_file_dir is not None:
            save_quant_data(output, quant_data_file_dir, f"{self.name}_q_y")

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
