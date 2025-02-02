import logging

import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.layernorm1d.design import (
    LayerNorm1d as LayerNorm1dDesign,
)
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


class LayerNorm1d(DesignCreatorModule, nn.LayerNorm):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            kwargs.get("norm_dim"), kwargs.get("eps"), kwargs.get("elementwise_affine")
        )

        self.device = kwargs.get("device")
        self.norm_dim = kwargs.get("norm_dim")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.input_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.normed_input_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.output_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)

    def create_design(self, name: str) -> LayerNorm1dDesign:
        return LayerNorm1dDesign(name=name)

    def _quant_weight(
        self, weight: torch.FloatTensor, weight_QParams: torch.nn.Module
    ) -> torch.IntTensor:
        quantised_weight = weight_QParams.quantizeProcess(weight)
        lower_bound = -(2 ** (self.quant_bits - 1))
        upper_bound = (2 ** (self.quant_bits - 1)) - 1

        if weight_QParams.is_symmetric == False:
            quantised_weight -= weight_QParams.zero_point
            lower_bound = -(2**self.quant_bits)
            upper_bound = (2**self.quant_bits) - 1

        return quantised_weight

    def _quant_bias(
        self, bias: torch.FloatTensor, bias_QParams: torch.nn.Module
    ) -> torch.IntTensor:
        quantised_bias = bias_QParams.quantizeProcess(bias)
        lower_bound = -(2 ** (self.quant_bits - 1))
        upper_bound = (2 ** (self.quant_bits - 1)) - 1

        if bias_QParams.is_symmetric == False:
            quantised_bias -= bias_QParams.zero_point
            lower_bound = -(2**self.quant_bits)
            upper_bound = (2**self.quant_bits) - 1

    def precompute(self) -> None:
        self.quantised_weight = self._quant_weight(
            weight=self.weight.to(self.device), weight_QParams=self.weight_QParams
        )

        self.quantised_bias = self._quant_bias(
            bias=self.bias.to(self.device), bias_QParams=self.bias_QParams
        )

        self.m_1 = (
            self.normed_input_QParams.scale
            * self.weight_QParams.scale
            / self.output_QParams.scale
        )
        self.m_2 = self.bias_QParams.scale / self.output_QParams.scale

        self.m_1_N_shifts, self.m_1_int = scaling_M(self.m_1)
        self.m_2_N_shifts, self.m_2_int = scaling_M(self.m_2)

    def _int_sqrt_operation(self, n: torch.Tensor) -> torch.Tensor:
        assert torch.all(n >= 0), "All elements in n must be non-negative."
        result = torch.zeros_like(n, dtype=torch.int32)

        for idx in range(n.numel()):
            value = n.view(-1)[idx].item()

            if value == 0:
                continue

            x = value // 2 + 1

            while True:
                if x != 0:
                    new_x = (x + value // x) // 2
                    if new_x >= x:
                        break
                else:
                    x = 1
                x = new_x

            result.view(-1)[idx] = x
        return result

    def int_forward(
        self, input: torch.FloatTensor, do_dequant_output: torch.bool
    ) -> torch.FloatTensor:
        q_input = (
            self.input_QParams.quantizeProcess(input)
            if input.dtype == torch.FloatTensor
            else input
        )

        numerator = q_input - torch.sum(q_input, dim=-1, keepdim=True) / self.norm_dim
        numerator = numerator.to(torch.int32)
        var_input = (numerator**2).sum(dim=-1, keepdim=True) / self.norm_dim
        denominator = torch.sqrt(var_input).to(torch.int32)
        normed_input = numerator / denominator

        # prepare input 1
        if normed_input.dtype == torch.FloatTensor:
            q_input = self.normed_input_QParams.quantizeProcess(normed_input)

        q_input = q_input - self.normed_input_QParams.zero_point

        tmp_input1 = simulate_bitshifting(
            q_input * self.quantised_weight, self.m_1_N_shifts, self.m_1_int
        )

        # prepare input 2
        tmp_input2 = simulate_bitshifting(
            self.quantised_bias, self.m_2_N_shifts, self.m_2_int
        )

        # execute integer-only addition
        tmp = tmp_input1 + tmp_input2

        # process output
        output = tmp + self.output_QParams.zero_point.to("cpu")
        output = output.clamp(
            min=-(2 ** (self.quant_bits - 1)), max=(2 ** (self.quant_bits - 1)) - 1
        )

        if do_dequant_output:
            output = self.output_QParams.dequantizeProcess(output.to(self.device))

        return output.to(self.device)

    def forward(
        self, input: torch.FloatTensor, given_input_QParams: object
    ) -> torch.FloatTensor:
        assert input.dtype == torch.FloatTensor, "input must be a torch.FloatTensor"
        if self.training:
            if given_input_QParams is None:
                self.input_QParams.update_quant_params(input)
            else:
                self.input_QParams = given_input_QParams

            self.weight_QParams.update_quant_params(self.weight)
            self.bias_QParams.update_quant_params(self.bias)

        input = SimQuant.apply(input, self.input_QParams)

        mean_input = torch.sum(input, dim=-1, keepdim=True) / self.norm_dim
        numerator = input - mean_input
        var_input = (numerator**2).sum(dim=-1, keepdim=True) / self.norm_dim
        denominator = torch.sqrt(var_input)

        normed_input = numerator / denominator

        if self.training:
            self.normed_input_QParams.update_quant_params(normed_input)
        normed_input = SimQuant.apply(normed_input, self.normed_input_QParams)

        weight = SimQuant.apply(self.weight.to(self.device), self.weight_QParams)
        bias = SimQuant.apply(self.bias.to(self.device), self.bias_QParams)

        output = normed_input * weight + bias

        if self.training:
            self.output_QParams.update_quant_params(output)

        output = SimQuant.apply(output, self.output_QParams)

        assert output.dtype == torch.FloatTensor, "output must be a torch.FloatTensor"

        return output
