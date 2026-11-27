from typing import Literal

import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.MPQ.batchnorm1d.design import (
    BatchNorm1d as BatchNorm1dDesign,
)
from elasticai.creator.nn.integer.quant_utils import (
    GlobalMinMaxObserver,
    MPQSupport,
    SimQuant,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricSignedQParams,
    QParams,
)
from elasticai.creator.nn.integer.quant_utils.parameter_quantization import (
    accumulator_qparams,
    fold_batchnorm_affine,
    quantize_bias,
    quantize_weights,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class BatchNorm1d(DesignCreatorModule, nn.BatchNorm1d, MPQSupport):
    def __init__(
        self,
        *,
        name: str,
        norm_dim: int,
        eps: float,
        momentum: float,
        affine: bool,
        track_running_stats: bool,
        num_dimensions: int,
        in_features: int,
        out_features: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
        enable_error_analysis: bool = False,
        MPQ_strategy: Literal["inheritance", "requantizer"] | None = None,
    ):
        super().__init__(
            norm_dim,
            eps,
            momentum,
            affine,
            track_running_stats,
        )
        self.num_dimensions = num_dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.norm_dim = norm_dim

        self.name = name
        self.quantizable_elements = [
            "inputs",
            "weights",
            "outputs",
        ]
        self.quant_bits_per_element = None
        self.quant_data_dir = quant_data_dir
        self.device = device
        self.enable_error_analysis = enable_error_analysis

        self.math_ops = MathOperations()
        self._init_mpq_attributes(MPQ_strategy=MPQ_strategy)
        self.precomputed = False

    def set_quant_bits_from_config(self, quant_configs):
        quant_bits_per_element = {}
        for element in self.quantizable_elements:
            key = f"{self.name}.{element}"
            quant_bits_per_element[element] = quant_configs.get(key)

        self.quant_bits_per_element = quant_bits_per_element
        self._init_element_Qparams()

    def _init_element_Qparams(self):
        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["weights"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.modified_weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["weights"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["outputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def create_design(self, name: str) -> BatchNorm1dDesign:
        return BatchNorm1dDesign(
            name=name,
            x_data_width=self.inputs_QParams.quant_bits,
            w_data_width=self.weight_QParams.quant_bits,
            b_data_width=self.accumulator_width,
            accumulator_width=self.accumulator_width,
            y_data_width=self.outputs_QParams.quant_bits,
            num_dimensions=self.num_dimensions,
            in_features=self.in_features,
            out_features=self.out_features,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.modified_weight_QParams.zero_point.item(),
            z_b=0,
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(
        self, weight: torch.Tensor, weight_QParams: QParams
    ) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        q_weights = weight_QParams.quantize(weight).to("cpu")

        if not weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                weight_QParams.zero_point,
                weight_QParams.quant_bits + 1,
            )
        return q_weights

    def precompute(self):
        mean = self.running_mean
        var = self.running_var
        std = torch.sqrt(var + self.eps)

        weight = SimQuant.apply(self.weight, self.weight_QParams)
        modified_weight, modified_bias = fold_batchnorm_affine(
            weight, self.bias, mean, std
        )
        self.modified_weight_QParams.update_quant_params(modified_weight)
        self.q_weights = self._get_quantized_weights(
            modified_weight, self.modified_weight_QParams
        )

        accumulator = accumulator_qparams(
            self.inputs_QParams,
            self.modified_weight_QParams,
        )
        self.accumulator_width = accumulator.quant_bits
        self.q_bias = quantize_bias(modified_bias, accumulator)

        self.scale_factor_M = accumulator.scale_factor / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )
        self._precompute_requantizer_params()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self._apply_requantizer(q_inputs, "inputs")

        q_inputs = self.math_ops.intsub(
            q_inputs,
            self.inputs_QParams.zero_point,
            self.inputs_QParams.quant_bits + 1,
        )
        # integer-only
        tmp = self.math_ops.int_elementwise_mul(
            self.q_weights, q_inputs, self.accumulator_width
        )

        tmp = self.math_ops.intadd(tmp, self.q_bias, self.accumulator_width)
        # print("tmp + self.q_bias: ", self.q_bias)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

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
        inputs: torch.Tensor,
        given_inputs_QParams: object,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        if enable_simquant:
            if self.training:
                self._handle_input_QParams(
                    inputs,
                    given_inputs_QParams,
                    "inputs_QParams",
                    "prev_inputs_QParams",
                )

            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        if self.training:
            mean = inputs.mean(dim=(0, 1), keepdim=False)
            var = inputs.var(dim=(0, 1), keepdim=False, unbiased=False)
            n = inputs.shape[0] * inputs.shape[1]
            running_var = var * n / (n - 1)

            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * running_var
        else:
            mean = self.running_mean
            var = self.running_var

        std = torch.sqrt(var + self.eps)

        if enable_simquant:
            if self.training:
                self.weight_QParams.update_quant_params(self.weight)

            weight = SimQuant.apply(self.weight, self.weight_QParams)
            # Fake-quantize the folded affine operation used by integer inference.
            modified_weight, modified_bias = fold_batchnorm_affine(
                weight, self.bias, mean, std
            )
            if self.training:
                self.modified_weight_QParams.update_quant_params(modified_weight)
            modified_weight = SimQuant.apply(modified_weight, self.modified_weight_QParams)
            accumulator = accumulator_qparams(
                self.inputs_QParams,
                self.modified_weight_QParams,
            )
            modified_bias = SimQuant.apply(modified_bias, accumulator)
            outputs = inputs * modified_weight + modified_bias
        else:
            outputs = self.weight * ((inputs - mean) / std) + self.bias

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
