import torch
import torch.nn as nn

from elasticai.creator.nn.integer.concatenate.design import (
    Concatenate as ConcatenateDesign,
)
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Concatenate(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.inputs_size = kwargs.get("inputs_size")  # num_features of input1
        self.hidden_size = kwargs.get("hidden_size")  # num_features of input2
        self.num_dimensions = kwargs.get("num_dimensions")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

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

    def create_design(self, name) -> ConcatenateDesign:
        return ConcatenateDesign(
            name=name,
            data_width=self.quant_bits,
            inputs_size=self.inputs_size,
            hidden_size=self.hidden_size,
            num_dimensions=self.num_dimensions,
            m_q_1=self.scale_factor_m_q_1.item(),
            m_q_2=self.scale_factor_m_q_2.item(),
            m_q_1_shift=self.scale_factor_m_q_1_shift.item(),
            m_q_2_shift=self.scale_factor_m_q_2_shift.item(),
            z_x1=self.inputs1_QParams.zero_point.item(),
            z_x2=self.inputs2_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def precompute(self):
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
        self,
        q_inputs1: torch.IntTensor,  # inputs
        q_inputs2: torch.IntTensor,  # h_prev
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs1, self.quant_data_dir, f"{self.name}_q_x_1")
        save_quant_data(q_inputs2, self.quant_data_dir, f"{self.name}_q_x_2")
        print("---------------------------")
        print("concatenate x1", q_inputs1)
        print("concatenate x2", q_inputs2)

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

        q_outputs = torch.cat((q_inputs1, q_inputs2), dim=self.num_dimensions)
        q_outputs = self.math_ops.intadd(
            q_outputs, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        print("concatenate y", q_outputs)

        return q_outputs

    def forward(
        self,
        inputs1: torch.FloatTensor,  # inputs
        inputs2: torch.FloatTensor,  # h_prev
        given_inputs1_QParams: torch.nn.Module = None,
        given_inputs2_QParams: torch.nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
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

        outputs = torch.cat((inputs1, inputs2), dim=self.num_dimensions)

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)

            outputs = SimQuant.apply(outputs, self.outputs_QParams)

        return outputs
