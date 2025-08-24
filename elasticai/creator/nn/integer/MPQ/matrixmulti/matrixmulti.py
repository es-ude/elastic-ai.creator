import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.MPQ.matrixmulti.design import (
    MatrixMulti as MatrixMultiDesign,
)
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    MPQSupport,
    SimQuant,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class MatrixMulti(DesignCreatorModule, nn.Module, MPQSupport):
    def __init__(self, **kwargs):
        super().__init__()

        self.x_1_dim_a = kwargs.get("x_1_dim_a")
        self.x_1_dim_b = kwargs.get("x_1_dim_b")
        self.x_1_dim_c = kwargs.get("x_1_dim_c")
        self.x_2_dim_a = kwargs.get("x_2_dim_a")
        self.x_2_dim_b = kwargs.get("x_2_dim_b")
        self.x_2_dim_c = kwargs.get("x_2_dim_c")
        self.y_dim_a = kwargs.get("y_dim_a")
        self.y_dim_b = kwargs.get("y_dim_b")
        self.y_dim_c = kwargs.get("y_dim_c")

        # attention score, or attention context
        self.operation_mode = kwargs.get("operation_mode")
        self.additional_scale = kwargs.get("additional_scale")

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs1", "inputs2", "outputs"]
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
        self._init_Qparams()

    def _init_element_Qparams(self):
        self.inputs1_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs1"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.inputs2_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs2"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["outputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def create_design(self, name: str) -> MatrixMultiDesign:
        return MatrixMultiDesign(
            name=name,
            is_score_mode="true" if self.operation_mode == "score" else "false",
            x_1_data_width=self.quant_bits_per_element["inputs1"],
            x_2_data_width=self.quant_bits_per_element["inputs2"],
            y_data_width=self.quant_bits_per_element["outputs"],
            x_1_dim_a=self.x_1_dim_a,
            x_1_dim_b=self.x_1_dim_b,
            x_1_dim_c=self.x_1_dim_c,
            x_2_dim_a=self.x_2_dim_a,
            x_2_dim_b=self.x_2_dim_b,
            x_2_dim_c=self.x_2_dim_c,
            y_dim_a=self.y_dim_a,
            y_dim_b=self.y_dim_b,
            y_dim_c=self.y_dim_c,
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x_1=self.inputs1_QParams.zero_point.item(),
            z_x_2=self.inputs2_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def precompute(self) -> None:
        if self.additional_scale is not None:  # 1/sqrt(d_model)
            self.scale_factor_M = (
                self.inputs1_QParams.scale_factor
                * self.inputs2_QParams.scale_factor
                / (self.outputs_QParams.scale_factor * self.additional_scale)
            )
        else:
            self.scale_factor_M = (
                self.inputs1_QParams.scale_factor
                * self.inputs2_QParams.scale_factor
                / self.outputs_QParams.scale_factor
            )

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )
        self._precompute_requantizer_params()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs1: torch.IntTensor,
        q_inputs2: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs1, self.quant_data_dir, f"{self.name}_q_x_1")
        save_quant_data(q_inputs2, self.quant_data_dir, f"{self.name}_q_x_2")

        q_inputs1 = self._apply_requantizer(q_inputs1, "inputs1")
        q_inputs2 = self._apply_requantizer(q_inputs2, "inputs2")

        q_inputs1 = self.math_ops.intsub(
            q_inputs1,
            self.inputs1_QParams.zero_point,
            self.inputs1_QParams.quant_bits + 1,
        )

        q_inputs2 = self.math_ops.intsub(
            q_inputs2,
            self.inputs2_QParams.zero_point,
            self.inputs2_QParams.quant_bits + 1,
        )

        # integer-only matrix multiplication on CPU
        tmp = self.math_ops.int_matmul_4d(
            inputs1=q_inputs1,
            inputs2=q_inputs2,
            operation_mode=self.operation_mode,
            outputs_quant_bits=(self.quant_bits_per_element["outputs"] + 1) * 2,
        )

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        ).to("cpu")

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
        inputs1: torch.FloatTensor,
        inputs2: torch.FloatTensor,
        given_inputs1_QParams: object,
        given_inputs2_QParams: object,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        assert inputs1.ndim == 4, "Input must be a 4D tensor"
        assert inputs2.ndim == 4, "Input must be a 4D tensor"
        if enable_simquant:
            if self.training:
                self._handle_input_QParams(
                    inputs1,
                    given_inputs1_QParams,
                    "inputs1_QParams",
                    "prev_inputs1_QParams",
                )
                self._handle_input_QParams(
                    inputs2,
                    given_inputs2_QParams,
                    "inputs2_QParams",
                    "prev_inputs2_QParams",
                )

            inputs1 = SimQuant.apply(inputs1, self.inputs1_QParams)
            inputs2 = SimQuant.apply(inputs2, self.inputs2_QParams)

        outputs = self.math_ops.matmul_4d(
            inputs1=inputs1, inputs2=inputs2, operation_mode=self.operation_mode
        )

        if self.additional_scale is not None:
            outputs = outputs * self.additional_scale

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
