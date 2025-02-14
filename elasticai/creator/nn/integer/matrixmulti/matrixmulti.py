import logging

import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.matrixmulti.design import (
    MatrixMulti as MatrixMultiDesign,
)
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


class MatrixMulti(DesignCreatorModule, nn.Module):
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

        device = kwargs.get("device")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.operation_mode = kwargs.get("operation_mode")
        self.additional_scale = kwargs.get("additional_scale")
        self.quant_data_file_dir = kwargs.get("quant_data_file_dir")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputs1_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs2_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> MatrixMultiDesign:
        return MatrixMultiDesign(
            name=name,
            data_width=self.quant_bits,
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

    def int_forward(
        self,
        q_inputs1: torch.IntTensor,
        q_inputs2: torch.IntTensor,
    ) -> torch.IntTensor:
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
            outputs_quant_bits=(self.quant_bits + 1) * 2,
        )

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        ).to("cpu")

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        return q_outputs

    def forward(
        self,
        inputs1: torch.FloatTensor,
        inputs2: torch.FloatTensor,
        given_inputs1_QParams: object,
        given_inputs2_QParams: object,
    ) -> torch.FloatTensor:
        assert inputs1.ndim == 4, "Input must be a 4D tensor"
        assert inputs2.ndim == 4, "Input must be a 4D tensor"

        if self.training:
            if given_inputs1_QParams is None:
                self.inputs1_QParams.update_quant_params(inputs1)
            else:
                self.inputs1_QParams = given_inputs1_QParams

            if given_inputs2_QParams is None:
                self.inputs2_QParams.update_quant_params(inputs2)
            else:
                self.inputs2_QParams = given_inputs2_QParams

        inputs1 = SimQuant.apply(inputs1, self.inputs1_QParams)
        inputs2 = SimQuant.apply(inputs2, self.inputs2_QParams)

        outputs = self.math_ops.matmul_4d(
            inputs1=inputs1, inputs2=inputs2, operation_mode=self.operation_mode
        )

        if self.additional_scale is not None:
            outputs = outputs * self.additional_scale

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)

        outputs = SimQuant.apply(outputs, self.outputs_QParams)

        return outputs
