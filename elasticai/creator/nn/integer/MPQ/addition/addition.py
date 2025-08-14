import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.MPQ.addition.design import Addition as AdditionDesign
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


class Addition(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_features = kwargs.get("num_features")
        self.num_dimensions = kwargs.get("num_dimensions")

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs1", "inputs2", "outputs"]
        self.quant_bits_per_element = None
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        self.device = kwargs.get("device")

        self.use_pipeline_template = kwargs.get("use_pipeline_template", False)
        self.unroll_factor = kwargs.get("unroll_factor", 1)
        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.math_ops = MathOperations()
        self.precomputed = False

    def set_quant_bits_from_config(self, quant_configs):
        quant_bits_per_element = {}
        for element in self.quantizable_elements:
            key = f"{self.name}.{element}"
            quant_bits_per_element[element] = quant_configs.get(key)
        self.quant_bits_per_element = quant_bits_per_element
        self._init_element_Qparams()

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

    def create_design(self, name: str) -> AdditionDesign:
        return AdditionDesign(
            name=name,
            data_width=self.quant_bits_per_element["inputs1"],  # TODO
            num_features=self.num_features,
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
            use_pipeline_template=self.use_pipeline_template,
            unroll_factor=self.unroll_factor,
        )

    def precompute(self) -> None:
        self.scale_factor_M_1 = (
            self.inputs1_QParams.scale_factor / self.outputs_QParams.scale_factor
        )
        self.scale_factor_M_2 = (
            self.inputs2_QParams.scale_factor / self.outputs_QParams.scale_factor
        )
        # error_threshold = 10 ** (-(self.quant_bits - 2))
        # self.scale_factor_m_q_1_shift, self.scale_factor_m_q_1 = scaling_M(
        #     self.scale_factor_M_1, error_threshold
        # )
        # self.scale_factor_m_q_2_shift, self.scale_factor_m_q_2 = scaling_M(
        #     self.scale_factor_M_2, error_threshold
        # )

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

        save_quant_data(q_inputs1, self.quant_data_dir, f"{self.name}_q_x_1")
        save_quant_data(q_inputs2, self.quant_data_dir, f"{self.name}_q_x_2")

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

        outputs = inputs1 + inputs2

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
