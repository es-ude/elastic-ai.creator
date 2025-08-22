import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.MPQ.softmax.design import (
    SoftmaxLUT as SoftmaxLUTDesign,
)
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    MPQSupport,
    SimQuant,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class SoftmaxLUT(DesignCreatorModule, nn.Module, MPQSupport):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.dim_a = kwargs.get("dim_a")
        self.dim_b = kwargs.get("dim_b")
        self.dim_c = kwargs.get("dim_c")
        self.nhead = kwargs.get("nhead")
        self.window_size = kwargs.get("window_size")

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs", "outputs"]
        self.quant_bits_per_element = None
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")
        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.device = device
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
        self.inputs1_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.inputs2_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["outputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def create_design(self, name: str) -> SoftmaxLUTDesign:
        if self.quant_bits_per_element["inputs"] <= 4:
            numerator_out_data_width = self.quant_bits_per_element["inputs"] * 3 - 1
            denominator_out_data_width = self.quant_bits_per_element["inputs"] * 3
        else:
            numerator_out_data_width = self.quant_bits_per_element["inputs"] * 3 - 1
            denominator_out_data_width = self.quant_bits_per_element["inputs"] * 2
        return SoftmaxLUTDesign(
            name=name,
            data_width=self.quant_bits_per_element["inputs"],  # TODO
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_c=self.dim_c,
            numberator_lut_out_data_width=numerator_out_data_width,
            denominator_lut_out_data_width=denominator_out_data_width,
            z_x=self.inputs2_QParams.zero_point.item(),
            z_t=self.tmp_zero_point.item(),  # self.inputs2_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def precompute(self):
        # 1. Enumerate all possible integer inputs
        min_quant = self.inputs2_QParams.min_quant.item()
        max_quant = self.inputs2_QParams.max_quant.item()
        q_inputs = torch.arange(min_quant, max_quant + 1, 1).to("cpu")

        # 2. Dequantize the q_inputs into float inputs by s_x * (q_x - zero_point)
        q_inputs = q_inputs.to(self.inputs2_QParams.zero_point.device)
        q_inputs_offset = q_inputs - self.inputs2_QParams.zero_point
        inputs = q_inputs_offset * self.inputs2_QParams.scale_factor

        # 3. Get exponential of inputs
        tmp = torch.exp(inputs)  # exp(s_x * (q_x - zero_point))

        # 4. Identify quant_bits of tmp
        tmp_quant_bits = (
            3 * self.quant_bits_per_element["inputs"]
            if self.quant_bits_per_element["inputs"] <= 4
            else 2 * self.quant_bits_per_element["inputs"]
        )
        # get scale factor of tmp
        tmp_scale_factor = (1.0 - 0.0) / (
            ((2 ** (tmp_quant_bits - 1) - 1) - (-(2 ** (tmp_quant_bits - 1))))
            // ((self.dim_b * self.dim_c) * self.nhead)  # self.dim_c
        )
        # get zero point of tmp
        tmp_zero_point = (2 ** (tmp_quant_bits - 1) - 1) - (1.0 / tmp_scale_factor)
        tmp_zero_point = torch.tensor(tmp_zero_point, dtype=torch.int32)
        self.tmp_zero_point = tmp_zero_point.round_().clamp(
            -(2 ** (tmp_quant_bits - 1)), 2 ** (tmp_quant_bits - 1) - 1
        )

        # 5. Quantize tmp
        q_tmp = (tmp / tmp_scale_factor + self.tmp_zero_point).to(torch.int32)
        q_tmp = q_tmp.round_().clamp(
            -(2 ** (tmp_quant_bits - 1)), 2 ** (tmp_quant_bits - 1) - 1
        )

        # 6. Create LUTs for denominator
        self.Qinput2QDenominator_LUT_dict = {}
        for i in range(len(q_tmp)):
            self.Qinput2QDenominator_LUT_dict[q_inputs[i].item()] = int(q_tmp[i].item())

        # 7. Create LUTs for numerator
        q_numerator = tmp / (tmp_scale_factor * self.outputs_QParams.scale_factor)
        q_numerator = q_numerator.to(torch.int32)
        q_numerator = q_numerator.round_().clamp(
            -(2 ** (self.quant_bits_per_element["inputs"] * 3 - 1)),
            2 ** (self.quant_bits_per_element["inputs"] * 3 - 1) - 1,
        )
        self.Qinput2QNumerator_LUT_dict = {}
        for i in range(len(q_numerator)):
            self.Qinput2QNumerator_LUT_dict[q_inputs[i].item()] = int(
                q_numerator[i].item()
            )

        # 8. Speeding up lookup by creating a mapping tensor
        # define the range of indices
        min_idx = min(self.Qinput2QNumerator_LUT_dict.keys())
        max_idx = max(self.Qinput2QNumerator_LUT_dict.keys())
        # create the mapping tensors
        numerator_mapping = torch.zeros(max_idx - min_idx + 1, dtype=torch.int32)
        denominator_mapping = torch.zeros(max_idx - min_idx + 1, dtype=torch.int32)
        # fill the mapping tensors
        for k, v in self.Qinput2QNumerator_LUT_dict.items():
            numerator_mapping[k - min_idx] = v
        for k, v in self.Qinput2QDenominator_LUT_dict.items():
            denominator_mapping[k - min_idx] = v
        # move the mapping tensors to the device
        self.numerator_mapping = numerator_mapping.to(self.device)
        self.denominator_mapping = denominator_mapping.to(self.device)
        self.mapping_offset = min_idx

        self._precompute_requantizer_params()
        self.precomputed = True

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_inputs = self._apply_requantizer(q_inputs, "inputs")

        # 1. Compute q_inputs
        max_q_inputs = q_inputs.max(dim=-1, keepdim=True)[0]
        q_inputs = self.math_ops.intsub(
            q_inputs, max_q_inputs, self.inputs2_QParams.quant_bits + 1
        )
        q_inputs = self.math_ops.intadd(
            q_inputs,
            self.inputs2_QParams.zero_point,
            self.inputs2_QParams.quant_bits + 1,
        )

        # q_numerator = q_inputs.clone()  # tmp / (S_soft * S_t)
        # for i in range(q_inputs.shape[0]):
        #     for j in range(q_inputs.shape[1]):
        #         for k in range(q_inputs.shape[2]):
        #             for l in range(q_inputs.shape[3]):
        #                 q_numerator[i][j][k][l] = self.Qinput2QNumerator_LUT_dict[
        #                     q_inputs[i][j][k][l].item()
        #                 ]
        # q_denominator = q_inputs.clone()
        # for i in range(q_inputs.shape[0]):
        #     for j in range(q_inputs.shape[1]):
        #         for k in range(q_inputs.shape[2]):
        #             for l in range(q_inputs.shape[3]):
        #                 q_denominator[i][j][k][l] = (
        #                     self.Qinput2QDenominator_LUT_dict[
        #                         q_inputs[i][j][k][l].item()
        #                     ]
        #                     - self.tmp_zero_point
        #                 )

        indices = q_inputs - self.mapping_offset
        q_numerator = self.numerator_mapping[indices]
        q_denominator = self.denominator_mapping[indices] - self.tmp_zero_point

        q_denominator = q_denominator.sum(dim=-1, keepdim=True)  # sum(Q_t - Z_t)

        tmp = self.math_ops.int_division(
            q_numerator, q_denominator, self.quant_bits_per_element["inputs"] + 1
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.quant_bits_per_element["inputs"]
        )
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")

        dq_outputs = self.outputs_QParams.dequantize(
            q_outputs
        )  # just for attention heatmap
        if self.enable_error_analysis:
            save_quant_data(
                dq_outputs,
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_outputs, dq_outputs

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
                    "inputs1_QParams",
                    "prev_inputs1_QParams",
                )

        max_input = inputs.max(dim=-1, keepdim=True)[0]
        inputs = inputs - max_input

        if enable_simquant:
            if self.training:
                self.inputs2_QParams.update_quant_params(inputs)

            inputs = SimQuant.apply(inputs, self.inputs2_QParams)

        outputs = torch.softmax(inputs, dim=-1)

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
