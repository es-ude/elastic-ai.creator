import logging

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition.design import Addition as AdditionDesign
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


class Addition(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_features = kwargs.get("num_features")
        self.num_dimensions = kwargs.get("num_dimensions")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = kwargs.get("device")

        self.inputs1_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.inputs2_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> AdditionDesign:
        return AdditionDesign(
            name=name,
            data_width=self.quant_bits,
            num_features=self.num_features,
            num_dimensions=self.num_dimensions,
            m_q_1=self.scale_factor_M_1.item(),
            m_q_2=self.scale_factor_M_2.item(),
            m_q_1_shift=self.scale_factor_m_q_1_shift.item(),
            m_q_2_shift=self.scale_factor_m_q_2_shift.item(),
            z_x1=self.inputs1_QParams.zero_point.item(),
            z_x2=self.inputs2_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def precompute(self) -> None:
        self.scale_factor_M_1 = (
            self.inputs1_QParams.scale_factor / self.outputs_QParams.scale_factor
        )
        self.scale_factor_M_2 = (
            self.inputs2_QParams.scale_factor / self.outputs_QParams.scale_factor
        )

        self.scale_factor_m_q_1_shift, self.scale_factor_m_q_1 = scaling_M(
            self.scale_factor_M_1
        )
        self.scale_factor_m_q_2_shift, self.scale_factor_m_q_2 = scaling_M(
            self.scale_factor_M_2
        )

        self.precomputed = True

    def int_forward(
        self, q_inputs1: torch.IntTensor, q_inputs2: torch.IntTensor
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        q_inputs1 = self.math_ops.intsub(
            q_inputs1,
            self.inputs1_QParams.zero_point,
            self.inputs1_QParams.quant_bits + 1,
        )
        q_inputs1 = simulate_bitshifting(
            q_inputs1, self.scale_factor_m_q_1_shift, self.scale_factor_m_q_1
        )

        q_inputs2 = self.math_ops.intsub(
            q_inputs2,
            self.inputs2_QParams.zero_point,
            self.inputs2_QParams.quant_bits + 1,
        )
        q_inputs2 = simulate_bitshifting(
            q_inputs2, self.scale_factor_m_q_2_shift, self.scale_factor_m_q_2
        )

        tmp = self.math_ops.intadd(
            q_inputs1,
            q_inputs2,
            self.inputs2_QParams.quant_bits + 2,
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        return q_outputs

    def forward(
        self,
        inputs1: torch.FloatTensor,
        inputs2: torch.FloatTensor,
        given_inputs1_QParams: torch.nn.Module = None,
        given_inputs2_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
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

        outputs = inputs1 + inputs2

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)

        outputs = SimQuant.apply(outputs, self.outputs_QParams)

        return outputs
