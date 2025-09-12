import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.layernorm1d.design import (
    LayerNorm1d as LayerNorm1dDesign,
)
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


class LayerNorm1d(DesignCreatorModule, nn.LayerNorm1d):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            kwargs.get("norm_dim"), kwargs.get("eps"), kwargs.get("elementwise_affine")
        )

        self.num_dimensions = kwargs.get("num_dimensions")
        self.in_features = kwargs.get("in_features")
        self.out_features = kwargs.get("out_features")
        self.norm_dim = kwargs.get("norm_dim")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")
        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.normed_inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> LayerNorm1dDesign:
        return LayerNorm1dDesign(name=name)

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

    def precompute(self) -> None:
        self.q_weights = self._get_quantized_weights(
            weight=self.weight, weight_QParams=self.weight_QParams
        )

        self.q_bias = self._get_quantized_bias(
            bias=self.bias, bias_QParams=self.bias_QParams
        )

        self.scale_factor_M_1 = (
            self.normed_inputs_QParams.scale_factor
            * self.weight_QParams.scale_factor
            / self.outputs_QParams.scale_factor
        )
        self.scale_factor_M_2 = (
            self.bias_QParams.scale_factor / self.outputs_QParams.scale_factor
        )

        self.scale_factor_m_1_q_shift, self.scale_factor_m_1_q = scaling_M(
            self.scale_factor_M_1
        )
        self.scale_factor_m_2_q_shift, self.scale_factor_m_2_q = scaling_M(
            self.scale_factor_M_2
        )
        self.precomputed = True

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

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs_sum = torch.sum(q_inputs, dim=-1, keepdim=True)
        q_inputs_sum_mean = q_inputs_sum / self.norm_dim
        numerator = self.math_ops.intsub(
            q_inputs, q_inputs_sum_mean, self.inputs_QParams.quant_bits + 1
        )

        var = (numerator**2).sum(dim=-1, keepdim=True) / self.norm_dim
        denominator = torch.sqrt(var).to(torch.int32)
        normed_inputs = numerator / denominator

        # prepare inputs 1
        if normed_inputs.dtype == torch.FloatTensor:
            q_inputs = self.normed_inputs_QParams.quantizeProcess(normed_inputs)

        q_inputs = q_inputs - self.normed_inputs_QParams.zero_point

        tmp_input1 = simulate_bitshifting(
            q_inputs * self.q_weights,
            self.scale_factor_m_1_q_shift,
            self.scale_factor_m_1_q,
        )

        # prepare inputs 2
        tmp_input2 = simulate_bitshifting(
            self.q_bias, self.scale_factor_m_2_q_shift, self.scale_factor_m_2_q
        )

        # execute integer-only addition
        tmp = tmp_input1 + tmp_input2

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        if self.enable_error_analysis:
            save_quant_data(
                self.outputs_QParams.dequantize(q_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: object,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        if enable_simquant:
            if self.training:
                self.weight_QParams.update_quant_params(self.weight)
                self.bias_QParams.update_quant_params(self.bias)

            weight = SimQuant.apply(self.weight, self.weight_QParams)
            bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        normed_inputs = (inputs - mean) / std
        outputs = weight * normed_inputs + bias

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)
            outputs = SimQuant.apply(outputs, self.outputs_QParams)
            if self.enable_error_analysis:
                save_quant_data(
                    outputs,
                    self.quant_data_dir,
                    f"{self.name}_y",
                )
        return outputs
