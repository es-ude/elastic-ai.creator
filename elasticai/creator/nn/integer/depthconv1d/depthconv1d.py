import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.depthconv1d.design import (
    DepthConv1d as DepthConv1dDesign,
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


class DepthConv1d(DesignCreatorModule, nn.Conv1d):
    def __init__(self, **kwargs):
        super().__init__(
            in_channels=kwargs.get("in_channels"),
            out_channels=kwargs.get("in_channels"),
            kernel_size=kwargs.get("kernel_size"),
            padding=kwargs.get("padding"),
            groups=kwargs.get("in_channels"),
            bias=True,
        )

        self.seq_len = kwargs.get("seq_len")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.device = kwargs.get("device")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> DepthConv1dDesign:
        return DepthConv1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_channels=self.in_channels,
            seq_len=self.seq_len,
            kernel_size=self.kernel_size[0],
            padding=self.padding,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.weight_QParams.zero_point.item(),
            z_b=self.bias_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(self) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = self.weight_QParams.quantize(self.weight).to("cpu")

        if not self.weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                self.weight_QParams.zero_point,
                self.weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(self) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        new_bias_quant_bits = (self.inputs_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )

        self.bias_QParams.set_scale_factor(new_bias_scale_factor)
        self.bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        self.bias_QParams.set_quant_range(new_bias_quant_bits)
        q_bias = self.bias_QParams.quantize(self.bias).to("cpu")
        if not self.bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, self.bias_QParams.zero_point, new_bias_quant_bits + 1
            )
        return q_bias

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"
        self.q_weights = self._get_quantized_weights()
        if self.bias is not None:
            self.q_bias = self._get_quantized_bias()

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
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        print(f"----------------------{self.name}----------------------")
        print("q_inputs", q_inputs)

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )
        print("q_inputs -z_x ", q_inputs)

        if self.padding != 0:
            if self.padding == "same":
                padding_len = self.kernel_size // 2
            else:
                padding_len = self.padding[0]

            q_inputs = F.pad(
                input=q_inputs,
                pad=(padding_len, padding_len),
                mode="constant",
                value=self.inputs_QParams.zero_point.item(),
            )

        print("q_inputs after padding", q_inputs)

        print("q_weights", self.q_weights)
        print("q_bias", self.q_bias)

        # TODO: Implement intmatmul or F.conv1d
        tmp = F.conv1d(
            q_inputs,
            self.q_weights,
            self.q_bias,
            padding=0,
            groups=self.groups,
        )
        print("tmp", tmp)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )
        print("tmp after bitshifting", tmp)

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )
        print("q_outputs", q_outputs)
        print("shape of q_outputs", q_outputs.shape)
        return q_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

            self.weight_QParams.update_quant_params(self.weight)
            if self.bias is not None:
                self.bias_QParams.update_quant_params(self.bias)

        inputs = SimQuant.apply(inputs, self.inputs_QParams)
        weight = SimQuant.apply(self.weight, self.weight_QParams)
        if self.bias is not None:
            bias = SimQuant.apply(self.bias, self.bias_QParams)

        if self.bias is not None:
            outputs = F.conv1d(
                inputs, weight, bias, padding=self.padding, groups=self.groups
            )
        else:
            outputs = F.conv1d(inputs, weight, padding=self.padding, groups=self.groups)

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)

        outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
