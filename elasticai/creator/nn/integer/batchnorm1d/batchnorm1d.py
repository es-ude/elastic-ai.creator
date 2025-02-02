import logging

import torch
from torch import nn

from elasticai.creator.nn.integer.batchnorm1d.design import (
    BatchNorm1d as BatchNorm1dDesign,
)
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
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


class BatchNorm1d(DesignCreatorModule, nn.BatchNorm1d):
    def __init__(self, **kwargs):
        super().__init__(
            kwargs.get("norm_dim"),
            kwargs.get("eps"),
            kwargs.get("momentum"),
            kwargs.get("affine"),
            kwargs.get("track_running_stats"),
        )

        self.num_dimensions = kwargs.get("num_dimensions")
        self.in_features = kwargs.get("in_features")
        self.out_features = kwargs.get("out_features")
        self.norm_dim = kwargs.get("norm_dim")

        device = kwargs.get("device")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> BatchNorm1dDesign:
        return BatchNorm1dDesign(name=name)

    def _get_quantized_weights(
        self, weight: torch.FloatTensor, weight_QParams: nn.Module
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = weight_QParams.quantize(weight).to("cpu")

        if not weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                weight_QParams.zero_point,
                weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(
        self, bias: torch.FloatTensor, bias_QParams: nn.Module
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        new_bias_quant_bits = (self.inputs_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        bias_QParams.set_scale_factor(new_bias_scale_factor)
        bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        bias_QParams.set_quant_range(new_bias_quant_bits)
        q_bias = bias_QParams.quantize(bias).to("cpu")
        if not bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, bias_QParams.zero_point, new_bias_quant_bits + 1
            )
        return q_bias

    def precompute(self):
        mean = self.running_mean
        var = self.running_var
        std = torch.sqrt(var + self.eps)

        weight = SimQuant.apply(self.weight, self.weight_QParams)
        modified_weight = weight / std

        bias = SimQuant.apply(self.bias, self.bias_QParams)
        self.modified_bias = bias - modified_weight * mean

        self.weight_QParams.update_quant_params(modified_weight)
        self.q_weight = self._get_quantized_weights(
            modified_weight, self.weight_QParams
        )

        self.tmp_quant_bits = (self.quant_bits + 1) * 2
        self.q_modified_bias = self._get_quantized_bias(
            bias=self.modified_bias,
            bias_QParams=self.bias_QParams,
        )

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        # integer-only
        tmp = self.math_ops.int_dotproduct(self.q_weight, q_inputs, self.tmp_quant_bits)
        tmp = self.math_ops.intadd(tmp, self.q_modified_bias, self.tmp_quant_bits + 1)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )
        return q_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: object
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

        inputs = SimQuant.apply(inputs, self.inputs_QParams)

        if self.training:
            mean = inputs.mean(dim=(0, 1), keepdim=False)
            var = inputs.var(dim=(0, 1), keepdim=False)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        std = torch.sqrt(var + self.eps)

        if self.training:
            self.weight_QParams.update_quant_params(self.weight)
            self.bias_QParams.update_quant_params(self.bias)

        weight = SimQuant.apply(self.weight, self.weight_QParams)
        bias = SimQuant.apply(self.bias, self.bias_QParams)

        normed_input = (inputs - mean) / std
        outputs = weight * normed_input + bias

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)
        outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
