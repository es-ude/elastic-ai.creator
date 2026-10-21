import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.accumulator import accumulator_bits
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.parameter_quantization import (
    quantize_bias,
    quantize_weights,
)
from elasticai.creator.nn.integer.sepconv1dbn.pointconv1dbn.design import (
    PointConv1dBN as PointConv1dBNDesign,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class PointConv1dBN(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias_bits = kwargs.get("bias_bits", 32)

        self.in_channels = kwargs.get("in_channels")
        self.out_channels = kwargs.get("out_channels")
        self.seq_len = kwargs.get("seq_len")

        self.name = kwargs.get("name")
        self.activation_bits = kwargs.get("activation_bits", kwargs.get("quant_bits"))
        self.weight_bits = kwargs.get(
            "weight_bits", kwargs.get("quant_bits", self.activation_bits)
        )
        self.quant_bits = self.activation_bits
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.bn1d = nn.BatchNorm1d(num_features=self.out_channels)

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.weight_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.activation_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.activation_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> PointConv1dBNDesign:
        return PointConv1dBNDesign(
            name=name,
            data_width=self.quant_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            seq_len=self.seq_len,
            weights=self.q_fused_weights.tolist(),
            bias=self.q_fused_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.weight_QParams.zero_point.item(),
            z_b=self.bias_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(self, weight) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        return quantize_weights(weight, self.weight_QParams)

    def _get_quantized_bias(self, bias) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )

        return quantize_bias(bias, self.bias_QParams, new_bias_scale_factor, self.bias_bits)

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"

        std = torch.sqrt(self.bn1d.running_var + self.bn1d.eps)

        gamma_ = self.bn1d.weight / std
        fused_weight = self.conv1d.weight * gamma_.reshape(-1, 1, 1)
        fused_bias = (
            gamma_ * self.conv1d.bias - gamma_ * self.bn1d.running_mean + self.bn1d.bias
        )

        self.q_fused_weights = self._get_quantized_weights(fused_weight)
        self.q_fused_bias = self._get_quantized_bias(fused_bias)

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )
        self.tmp_quant_bits = accumulator_bits(
            int(self.inputs_QParams.quant_bits)
            + int(self.weight_QParams.quant_bits)
            + 2,
            self.q_fused_weights.shape[1] * self.q_fused_weights.shape[2],
            self.q_fused_bias,
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

        if self.q_fused_bias is not None:
            tmp = F.conv1d(
                q_inputs.to(torch.int64),
                self.q_fused_weights.to(device=q_inputs.device, dtype=torch.int64),
                self.q_fused_bias.to(device=q_inputs.device, dtype=torch.int64),
                padding=self.conv1d.padding,
            )
            tmp = self.math_ops.clamp_result(tmp, self.tmp_quant_bits)
        else:
            tmp = F.conv1d(
                q_inputs.to(torch.int64),
                self.q_fused_weights.to(device=q_inputs.device, dtype=torch.int64),
                padding=self.conv1d.padding,
            )
            tmp = self.math_ops.clamp_result(tmp, self.tmp_quant_bits)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )
        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams
            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        if self.training:
            tmp_outputs = F.conv1d(
                inputs,
                self.conv1d.weight,
                self.conv1d.bias,
                padding=self.conv1d.padding,
            )
            mean = tmp_outputs.mean(dim=(0, 2))  # [batch_size, out_channels, seq_len]
            var = tmp_outputs.var(dim=(0, 2), unbiased=False)
            n = tmp_outputs.shape[0] * tmp_outputs.shape[2]
            running_var = var * n / (n - 1)
            with torch.no_grad():
                self.bn1d.running_mean = (
                    1 - self.bn1d.momentum
                ) * self.bn1d.running_mean + self.bn1d.momentum * mean
                self.bn1d.running_var = (
                    1 - self.bn1d.momentum
                ) * self.bn1d.running_var + self.bn1d.momentum * running_var
        else:
            mean = self.bn1d.running_mean
            var = self.bn1d.running_var

        std = torch.sqrt(var + self.bn1d.eps)

        gamma_ = self.bn1d.weight / std
        fused_weight = self.conv1d.weight * gamma_.reshape(-1, 1, 1)
        fused_bias = gamma_ * self.conv1d.bias - gamma_ * mean + self.bn1d.bias

        if enable_simquant:
            if self.training:
                self.weight_QParams.update_quant_params(fused_weight)
                self.bias_QParams.update_quant_params(fused_bias)

            fused_weight = SimQuant.apply(fused_weight, self.weight_QParams)
            fused_bias = SimQuant.apply(fused_bias, self.bias_QParams)

        outputs = F.conv1d(
            inputs, fused_weight, fused_bias, padding=self.conv1d.padding
        )

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)

            outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
