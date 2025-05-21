import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.linearrelu.design import Linear as LinearReLUDesign
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


class LinearReLU(DesignCreatorModule, nn.Linear):
    def __init__(self, **kwargs):
        super().__init__(
            kwargs.get("in_features"), kwargs.get("out_features"), kwargs.get("bias")
        )
        self.num_dimensions = kwargs.get("num_dimensions", None)

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        # TODO: quantization scheme for each quantiztaion objects should be chosen by the user
        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        if kwargs.get("bias"):
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

    def create_design(self, name: str) -> LinearReLUDesign:
        return LinearReLUDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            num_dimensions=self.num_dimensions,
            out_features=self.out_features,
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
        else:
            self.q_bias = None

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

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        tmp = self.math_ops.int_mac(
            x=q_inputs,
            w=self.q_weights,
            b=self.q_bias,
            x_quant_bits=self.inputs_QParams.quant_bits,
            w_quant_bits=self.weight_QParams.quant_bits,
        )

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        q_outputs = torch.maximum(
            q_outputs, self.outputs_QParams.zero_point.clone().detach()
        )  # ReLU

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")

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
            outputs = F.linear(inputs, weight, bias)
        else:
            outputs = F.linear(inputs, weight)

        outputs = F.relu(outputs)

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)

        outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
