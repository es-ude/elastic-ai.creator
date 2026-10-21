import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.conv1d.design import Conv1d as Conv1dDesign
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import (
    get_padded_q_inputs,
    get_padding_len,
)
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
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Conv1d(DesignCreatorModule, nn.Conv1d):
    def __init__(self, **kwargs):
        for option in ("stride", "dilation", "groups"):
            value = kwargs.get(option, 1)
            if value != 1 and not (option != "groups" and value == (1,)):
                raise ValueError(f"Conv1d only supports {option}=1")
        if kwargs.get("padding_mode", "zeros") != "zeros":
            raise ValueError("Conv1d only supports padding_mode='zeros'")

        super().__init__(
            kwargs.get("in_channels"),
            kwargs.get("out_channels"),
            kwargs.get("kernel_size"),
            padding=kwargs.get("padding", 0),
            bias=kwargs.get("bias", True),
            device=kwargs.get("device"),
        )
        self.bias_bits = kwargs.get("bias_bits", 32)
        self.seq_len = kwargs.get("seq_len")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.bias_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> Conv1dDesign:
        return Conv1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size[0],
            seq_len=self.seq_len,
            padding=self.padding,
            padding_len=self.padding_len,
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
        return quantize_weights(self.weight, self.weight_QParams)

    def _get_quantized_bias(self) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )
        return quantize_bias(self.bias, self.bias_QParams, new_bias_scale_factor, self.bias_bits)

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

        self.tmp_quant_bits = accumulator_bits(
            int(self.inputs_QParams.quant_bits)
            + int(self.weight_QParams.quant_bits)
            + 2,
            self.q_weights.shape[1] * self.q_weights.shape[2],
            self.q_bias,
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

        self.padding_len = get_padding_len(self.padding, self.kernel_size)
        q_inputs = get_padded_q_inputs(
            padding_len=self.padding_len,
            q_inputs=q_inputs,
            inputs_QParams=self.inputs_QParams,
        )

        bias = (
            self.q_bias.to(device=q_inputs.device, dtype=torch.int64)
            if self.q_bias is not None
            else None
        )

        tmp = F.conv1d(
            q_inputs.to(torch.int64),
            self.q_weights.to(device=q_inputs.device, dtype=torch.int64),
            bias=bias,
            padding=0,
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

                self.weight_QParams.update_quant_params(self.weight)
                if self.bias is not None:
                    self.bias_QParams.update_quant_params(self.bias)

            inputs = SimQuant.apply(inputs, self.inputs_QParams)
            weight = SimQuant.apply(self.weight, self.weight_QParams)
            if self.bias is not None:
                bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        if self.bias is not None:
            outputs = F.conv1d(inputs, weight, bias, padding=self.padding)
        else:
            outputs = F.conv1d(inputs, weight, padding=self.padding)

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)
            outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
