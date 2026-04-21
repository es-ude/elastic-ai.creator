import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.MPQ.avgpooling1dflatten.design import (
    AVGPooling1dFlatten as AVGPooling1dFlattenDesign,
)
from elasticai.creator.nn.integer.quant_utils import (
    GlobalMinMaxObserver,
    MPQSupport,
    SimQuant,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.MPQParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class AVGPooling1dFlatten(DesignCreatorModule, nn.Module, MPQSupport):
    def __init__(self, **kwargs):
        super().__init__()

        self.in_features = kwargs.get("in_features")
        self.out_features = kwargs.get("out_features")
        self.in_num_dimensions = kwargs.get("in_num_dimensions")
        self.out_num_dimensions = kwargs.get("out_num_dimensions")

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs", "outputs"]
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
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["outputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def create_design(self, name: str) -> AVGPooling1dFlattenDesign:
        return AVGPooling1dFlattenDesign(
            name=name,
            x_data_width=self.inputs_QParams.quant_bits.item(),
            y_data_width=self.outputs_QParams.quant_bits.item(),
            in_features=self.in_features,
            out_features=self.out_features,
            in_num_dimensions=self.in_num_dimensions,
            out_num_dimensions=self.out_num_dimensions,
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"
        L = self.in_features  # e.g., seq_len
        self.scale_factor_M = torch.tensor(
            self.inputs_QParams.scale_factor.item()
            / (self.outputs_QParams.scale_factor.item() * L),
        )
        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M.clone().detach()
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
        q_inputs = q_inputs.permute(0, 2, 1)

        q_inputs = self._apply_requantizer(q_inputs, "inputs")

        q_inputs = self.math_ops.intsub(
            q_inputs,
            self.inputs_QParams.zero_point,
            self.inputs_QParams.quant_bits.item() + 1,
        )

        # assume that (batch_size, channels, seq_len) is the shape of q_inputs
        tmp = torch.sum(q_inputs, dim=2, keepdim=True).to(
            torch.int32
        )  # sum over seq_len

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits.item()
        )
        q_outputs = q_outputs.squeeze(2)
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
            inputs = SimQuant.apply(inputs, self.inputs_QParams)

        inputs = inputs.permute(0, 2, 1)

        # assume that (batch_size, channels, seq_len) is the shape of inputs
        outputs = F.avg_pool1d(inputs, kernel_size=inputs.size(2))
        outputs = outputs.squeeze(2)

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
