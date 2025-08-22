import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.MPQ.linear.design import Linear as LinearDesign
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    MPQSupport,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Linear(DesignCreatorModule, nn.Linear, MPQSupport):
    def __init__(self, **kwargs):
        super().__init__(
            kwargs.get("in_features"), kwargs.get("out_features"), kwargs.get("bias")
        )
        self.num_dimensions = kwargs.get("num_dimensions", None)
        self.enable_bias = kwargs.get("bias", True)

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs", "weights", "bias", "outputs"]
        self.quant_bits_per_element = None
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        self.device = kwargs.get("device")
        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.math_ops = MathOperations()
        self._init_mpq_attributes(**kwargs)  # MPQ
        self.precomputed = False

    def set_quant_bits_from_config(self, quant_configs):
        quant_bits_per_element = {}
        for element in self.quantizable_elements:
            key = f"{self.name}.{element}"
            quant_bits_per_element[element] = quant_configs.get(key)
        self.quant_bits_per_element = quant_bits_per_element
        self._init_element_Qparams()

    def _init_element_Qparams(self):
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["weights"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

        if self.enable_bias:
            self.bias_QParams = SymmetricSignedQParams(
                quant_bits=self.quant_bits_per_element["bias"],
                observer=GlobalMinMaxObserver(),
            ).to(self.device)

        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["outputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def create_design(self, name: str) -> LinearDesign:
        return LinearDesign(
            name=name,
            data_width=self.quant_bits_per_element["inputs"],  # TODO
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

        self._precompute_requantizer_params()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self._apply_requantizer(q_inputs, "inputs")

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
        given_inputs_QParams: torch.nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                self._handle_input_QParams(
                    inputs,
                    given_inputs_QParams,
                    "inputs_QParams",
                    "prev_inputs_QParams",
                )

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

        outputs = (
            F.linear(inputs, weight, bias)
            if bias is not None
            else F.linear(inputs, weight)
        )

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
