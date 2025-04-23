import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.softmax.design import SoftmaxLUT as SoftmaxLUTDesign
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class SoftmaxLUT(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.dim_a = kwargs.get("dim_a")
        self.dim_b = kwargs.get("dim_b")
        self.dim_c = kwargs.get("dim_c")
        self.nhead = kwargs.get("nhead")
        self.window_size = kwargs.get("window_size")

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

    def create_design(self, name: str) -> SoftmaxLUTDesign:
        if self.quant_bits <= 4:
            numerator_out_data_width = self.quant_bits * 3 - 1
            denominator_out_data_width = self.quant_bits * 3
        else:
            numerator_out_data_width = self.quant_bits * 3 - 1
            denominator_out_data_width = self.quant_bits * 2
        return SoftmaxLUTDesign(
            name=name,
            data_width=self.quant_bits,
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
        # prepare all possible quantised input
        min_quant = self.inputs2_QParams.min_quant.item()
        max_quant = self.inputs2_QParams.max_quant.item()
        q_inputs = torch.arange(min_quant, max_quant + 1, 1).to("cpu")

        # subtract max_quant from q_inputs
        q_inputs_offset = (
            q_inputs.to(self.inputs2_QParams.zero_point.device)
            - self.inputs2_QParams.zero_point
        )  # q_x - zero_point
        inputs = (
            q_inputs_offset * self.inputs2_QParams.scale_factor
        )  # s_x * (q_x - zero_point)

        tmp = torch.exp(inputs)  # exp(s_x * (q_x - zero_point))

        tmp_quant_bits = (
            3 * self.quant_bits if self.quant_bits <= 4 else 2 * self.quant_bits
        )
        tmp_scale_factor = (1.0 - 0.0) / (
            ((2 ** (tmp_quant_bits - 1) - 1) - (-(2 ** (tmp_quant_bits - 1))))
            // ((self.window_size**2) * self.nhead)  # self.window_size
        )  # get scale_factor factor of tmp
        tmp_zero_point = (2 ** (tmp_quant_bits - 1) - 1) - (
            1.0 / tmp_scale_factor
        )  # get zero point of tmp
        tmp_zero_point = torch.tensor(tmp_zero_point, dtype=torch.int32)
        self.tmp_zero_point = tmp_zero_point.round_().clamp(
            -(2 ** (tmp_quant_bits - 1)), 2 ** (tmp_quant_bits - 1) - 1
        )
        q_tmp = (tmp / tmp_scale_factor + self.tmp_zero_point).to(torch.int32)
        q_tmp = q_tmp.round_().clamp(
            -(2 ** (tmp_quant_bits - 1)), 2 ** (tmp_quant_bits - 1) - 1
        )

        # for denominator
        self.Qinput2QDenominator_LUT_dict = {}
        for i in range(len(q_tmp)):
            self.Qinput2QDenominator_LUT_dict[q_inputs[i].item()] = int(q_tmp[i].item())

        # for numerator
        q_numerator = tmp / (tmp_scale_factor * self.outputs_QParams.scale_factor)
        q_numerator = q_numerator.to(torch.int32)
        q_numerator = q_numerator.round_().clamp(
            -(2 ** (self.quant_bits * 3 - 1)), 2 ** (self.quant_bits * 3 - 1) - 1
        )

        self.Qinput2QNumerator_LUT_dict = {}
        for i in range(len(q_numerator)):
            self.Qinput2QNumerator_LUT_dict[q_inputs[i].item()] = int(
                q_numerator[i].item()
            )
        self.precomputed = True

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        max_q_inputs = q_inputs.max(dim=-1, keepdim=True)[0]

        q_inputs = self.math_ops.intsub(
            q_inputs, max_q_inputs, self.inputs2_QParams.quant_bits + 1
        )

        q_inputs = self.math_ops.intadd(
            q_inputs,
            self.inputs2_QParams.zero_point,
            self.inputs2_QParams.quant_bits + 1,
        )

        q_numerator = q_inputs.clone()  # tmp / (S_soft * S_t)
        for i in range(q_inputs.shape[0]):
            for j in range(q_inputs.shape[1]):
                for k in range(q_inputs.shape[2]):
                    for l in range(q_inputs.shape[3]):
                        q_numerator[i][j][k][l] = self.Qinput2QNumerator_LUT_dict[
                            q_inputs[i][j][k][l].item()
                        ]
        q_denominator = q_inputs.clone()
        for i in range(q_inputs.shape[0]):
            for j in range(q_inputs.shape[1]):
                for k in range(q_inputs.shape[2]):
                    for l in range(q_inputs.shape[3]):
                        q_denominator[i][j][k][l] = (
                            self.Qinput2QDenominator_LUT_dict[
                                q_inputs[i][j][k][l].item()
                            ]
                            - self.tmp_zero_point
                        )
        q_denominator = q_denominator.sum(dim=-1, keepdim=True)  # sum(Q_t - Z_t)

        tmp = self.math_ops.int_division(
            q_numerator, q_denominator, self.quant_bits + 1
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.quant_bits
        )
        dq_outputs = self.outputs_QParams.dequantize(q_outputs)
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs, dq_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs1_QParams.update_quant_params(inputs)
            else:
                self.inputs1_QParams = given_inputs_QParams

        max_input = inputs.max(dim=-1, keepdim=True)[0]
        inputs = inputs - max_input

        if self.training:
            self.inputs2_QParams.update_quant_params(inputs)

        inputs = SimQuant.apply(inputs, self.inputs2_QParams)
        outputs = torch.softmax(inputs, dim=-1)

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)
        outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
