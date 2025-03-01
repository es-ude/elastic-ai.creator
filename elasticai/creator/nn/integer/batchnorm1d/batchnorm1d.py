import logging

import torch
from torch import nn

from elasticai.creator.nn.integer.batchnorm1d.design import (
    BatchNorm1d as BatchNorm1dDesign,
)
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
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

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.modified_weight_QParams = AsymmetricSignedQParams(
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
        return BatchNorm1dDesign(
            name=name,
            data_width=self.quant_bits,
            num_dimensions=self.num_dimensions,
            in_features=self.in_features,
            out_features=self.out_features,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.modified_weight_QParams.zero_point.item(),
            z_b=self.bias_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(
        self, weight: torch.FloatTensor, weight_QParams: nn.Module
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = weight_QParams.quantize(weight).to("cpu")

        if weight_QParams.is_symmetric == False:
            q_weights = self.math_ops.intsub(
                q_weights,
                weight_QParams.zero_point,
                weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(
        self,
        bias: torch.FloatTensor,
        bias_QParams: nn.Module,
        given_quant_scale: torch.FloatTensor,
        given_quant_bits: int,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"

        bias_QParams.set_scale_factor(given_quant_scale)
        bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        bias_QParams.set_quant_range(given_quant_bits)
        q_bias = bias_QParams.quantize(bias).to("cpu")
        return q_bias

    def precompute(self):
        mean = self.running_mean
        var = self.running_var
        std = torch.sqrt(var + self.eps)

        self.tmp_quant_bits = (self.weight_QParams.quant_bits + 1) + (
            self.inputs_QParams.quant_bits + 1
        )

        weight = SimQuant.apply(self.weight, self.weight_QParams)
        modified_weight = weight / std
        self.modified_weight_QParams.update_quant_params(modified_weight)
        self.q_weights = self._get_quantized_weights(
            modified_weight, self.modified_weight_QParams
        )

        bias = SimQuant.apply(self.bias, self.bias_QParams)
        modified_bias = bias - modified_weight * mean
        new_bias_QParms_scale = (
            self.inputs_QParams.scale_factor * self.modified_weight_QParams.scale_factor
        )
        self.q_bias = self._get_quantized_bias(
            bias=modified_bias,
            bias_QParams=self.bias_QParams,
            given_quant_scale=new_bias_QParms_scale,
            given_quant_bits=self.tmp_quant_bits,
        )

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.modified_weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )
        # integer-only
        tmp = self.math_ops.int_dotproduct(
            self.q_weights, q_inputs, self.tmp_quant_bits
        )

        tmp = self.math_ops.intadd(tmp, self.q_bias, self.tmp_quant_bits + 1)
        # print("tmp + self.q_bias: ", self.q_bias)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        # print("q_outputs: ", q_outputs)
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
