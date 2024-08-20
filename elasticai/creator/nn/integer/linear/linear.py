import logging

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.linear.design import Linear as LinearDesign
from elasticai.creator.nn.integer.math_operations.MathOperations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricSignedQParams,
    SymmetricSignedQParams,
)
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)
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
        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"), observer=GlobalMinMaxObserver()
        )
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"), observer=GlobalMinMaxObserver()
        )
        self.input_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"), observer=GlobalMinMaxObserver()
        )
        self.output_QParams = AsymmetricSignedQParams(
            quant_bits=kwargs.get("quant_bits"), observer=GlobalMinMaxObserver()
        )

        self.math_ops = MathOperations()

    def create_design(self, name: str) -> LinearDesign:
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

    def _get_quantized_weights(self) -> torch.IntTensor:
        q_weight = self.weight_QParams.quantize(self.weight)

        if not self.weight_QParams.is_symmetric:
            q_weight = self.math_ops.intsub(
                q_weight,
                self.weight_QParams.zero_point,
                self.weight_QParams.quant_bits + 1,
            )
        return q_weight

    def _get_quantized_bias(self) -> torch.IntTensor:
        new_bias_scale_factor = (
            self.input_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        new_bias_quant_bits = (self.input_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )

        self.bias_QParams.set_scale_factor(new_bias_scale_factor)
        self.bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        self.bias_QParams.set_quant_range(new_bias_quant_bits)

        q_bias = self.bias_QParams.quantize(self.bias)

        if not self.bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, self.bias_QParams.zero_point, new_bias_quant_bits + 1
            )
        return q_bias

    def precompute(self) -> None:
        self.q_weight = self._get_quantized_weights()
        self.q_bias = self._get_quantized_bias()

        self.tmp_quant_bits = self.bias_QParams.quant_bits

        self.scale_factor_M = (
            self.input_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.output_QParams.scale_factor

        self.scale_factor_M_q_shift, self.scale_factor_M_q = scaling_M(
            self.scale_factor_M
        )

    def int_forward(
        self,
        q_input: torch.IntTensor,
    ) -> torch.IntTensor:
        q_input = self.math_ops.intsub(
            q_input, self.input_QParams.zero_point, self.input_QParams.quant_bits + 1
        )

        tmp_quant_bits = (self.input_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        tmp = self.math_ops.intmatmul(
            q_input,
            self.q_weight.t(),
            tmp_quant_bits,  # TODO further +1 or not
        )

        if self.bias is not None:
            tmp = self.math_ops.intadd(tmp, self.q_bias, self.tmp_quant_bits + 1)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_M_q_shift, self.scale_factor_M_q
        )

        output = self.math_ops.intadd(
            tmp, self.output_QParams.zero_point, self.output_QParams.quant_bits
        )

        return output

    def forward(
        self, input: torch.FloatTensor, given_input_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_input_QParams is None:
                self.input_QParams.update_quant_params(input)
            else:
                self.input_QParams = given_input_QParams

        self.weight_QParams.update_quant_params(self.weight)
        self.bias_QParams.update_quant_params(self.bias)

        input = SimQuant.apply(input, self.input_QParams)
        weight = SimQuant.apply(self.weight, self.weight_QParams)
        bias = SimQuant.apply(self.bias, self.bias_QParams)

        output = F.linear(input, weight, bias)

        if self.training:
            self.output_QParams.update_quant_params(output)

        output = SimQuant.apply(output, self.output_QParams)
        return output
