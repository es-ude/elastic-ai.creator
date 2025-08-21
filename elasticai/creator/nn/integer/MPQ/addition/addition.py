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

        # MPQ related attributes
        self.MPQ_strategy = kwargs.get("MPQ_strategy")  # "inheritance", "requantizer"
        self.prev_inputs1_QParams = None
        self.prev_inputs2_QParams = None

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

        self.scale_factor_m_q_1_shift, self.scale_factor_m_q_1 = scaling_M(
            self.scale_factor_M_1
        )
        self.scale_factor_m_q_2_shift, self.scale_factor_m_q_2 = scaling_M(
            self.scale_factor_M_2
        )

        if self.MPQ_strategy == "requantizer":
            assert (
                self.prev_inputs1_QParams is not None
            ), "prev_inputs1_QParams must be set for requantizer strategy"
            assert (
                self.prev_inputs2_QParams is not None
            ), "prev_inputs2_QParams must be set for requantizer strategy"

            requantizer_M_1 = (
                self.prev_inputs1_QParams.scale_factor
                / self.inputs1_QParams.scale_factor
            )
            requantizer_M_2 = (
                self.prev_inputs2_QParams.scale_factor
                / self.inputs2_QParams.scale_factor
            )

            self.requantizer_m_q_1_shift, self.requantizer_m_q_1 = scaling_M(
                requantizer_M_1
            )

            self.requantizer_m_q_2_shift, self.requantizer_m_q_2 = scaling_M(
                requantizer_M_2
            )

        self.precomputed = True

    def _requantize_inputs(
        self,
        prev_q_inputs: torch.IntTensor,
        prev_QParams: object,
        m_q_1_shift: int,
        m_q_1: int,
    ) -> torch.IntTensor:
        # s_curr (q_curr - z_curr) = s_prev  (q_prev - z_prev)
        curr_q_inputs = self.math_ops.intsub(
            prev_q_inputs,
            prev_QParams.zero_point,
            prev_QParams.quant_bits + 1,
        )

        # (q_curr - z_curr) = s_prev/s_curr (q_prev - z_prev)
        curr_q_inputs = simulate_bitshifting(
            curr_q_inputs,
            m_q_1_shift,
            m_q_1,
        )

        # q_curr = s_prev/s_curr (q_prev - z_prev) + z_curr
        curr_q_inputs = self.math_ops.intadd(
            curr_q_inputs,
            self.inputs1_QParams.zero_point,
            self.inputs1_QParams.quant_bits + 1,
        )
        return curr_q_inputs

    def _handle_input_qparams(
        self,
        inputs: torch.FloatTensor,
        given_QParams: object,
        current_QParams_attr: str,
        prev_QParams_attr: str,
    ):
        if given_QParams is None:
            getattr(self, current_QParams_attr).update_quant_params(inputs)
            return

        if self.MPQ_strategy == "requantizer":
            getattr(self, current_QParams_attr).update_quant_params(inputs)
            if prev_QParams_attr is not None:
                setattr(self, prev_QParams_attr, given_QParams)
        elif self.MPQ_strategy == "inheritance":
            setattr(self, current_QParams_attr, given_QParams)
        else:
            raise ValueError(f"Unsupported MPQ strategy: {self.MPQ_strategy}")

    def _requantizer(self, q_inputs1: torch.IntTensor, q_inputs2: torch.IntTensor):
        q_inputs1 = self._requantize_inputs(
            prev_q_inputs=q_inputs1,
            prev_QParams=self.prev_inputs1_QParams,
            m_q_1_shift=self.requantizer_m_q_1_shift,
            m_q_1=self.requantizer_m_q_1,
        )
        q_inputs2 = self._requantize_inputs(
            prev_q_inputs=q_inputs2,
            prev_QParams=self.prev_inputs2_QParams,
            m_q_1_shift=self.requantizer_m_q_2_shift,
            m_q_1=self.requantizer_m_q_2,
        )
        return q_inputs1, q_inputs2

    def int_forward(
        self, q_inputs1: torch.IntTensor, q_inputs2: torch.IntTensor
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        if self.MPQ_strategy == "requantizer":
            return self._requantizer(q_inputs1, q_inputs2)

        save_quant_data(q_inputs1, self.quant_data_dir, f"{self.name}_q_x_1")
        save_quant_data(q_inputs2, self.quant_data_dir, f"{self.name}_q_x_2")

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

        q_inputs1 = simulate_bitshifting(
            q_inputs1, self.scale_factor_m_q_1_shift, self.scale_factor_m_q_1
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
                self._handle_input_qparams(
                    inputs1,
                    given_inputs1_QParams,
                    "inputs1_QParams",
                    "prev_inputs1_QParams",
                )
                self._handle_input_qparams(
                    inputs2,
                    given_inputs2_QParams,
                    "inputs2_QParams",
                    "prev_inputs2_QParams",
                )

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
