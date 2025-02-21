import logging
from pathlib import Path

import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.softmax.design import SoftmaxLUT as SoftmaxLUTDesign


class SoftmaxLUT(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.dim_a = kwargs.get("dim_a")
        self.dim_b = kwargs.get("dim_b")
        self.dim_c = kwargs.get("dim_c")
        self.nhead = kwargs.get("nhead")
        self.window_size = kwargs.get("window_size")

        self.device = kwargs.get("device")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

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
            z_x=self.inputs1_QParams.zero_point.item(),
            z_t=self.inputs2_QParams.zero_point.item(),
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _compute_scale_zero_point(self, quant_bits, divisor, t):
        max_int = 2 ** (quant_bits - 1) - 1
        s_t = (1.0 - 0.0) / ((2 * max_int) // divisor)
        z_t = max_int - (1.0 / s_t)

        z_t = torch.tensor(z_t).round_().clamp(-max_int, max_int)
        q_t = (t / s_t + z_t).to(dtype=torch.int32).round_().clamp(-max_int, max_int)

        return s_t, z_t, q_t

    def precompute(self):
        # prepare all possible quantised inputs
        min_quant = self.inputs2_QParams.min_quant.item()
        max_quant = self.inputs2_QParams.max_quant.item()
        q_inputs = torch.arange(min_quant, max_quant + 1, 1).to(self.device)  # q_x

        # subtract max_quant from q_inputs
        q_inputs_offset = self.math_ops.intsub(
            q_inputs.to(dtype=torch.int32),
            self.inputs2_QParams.zero_point.to("cpu"),
            self.inputs2_QParams.quant_bits + 1,
        )  # q_x - zero_point

        inputs = q_inputs_offset * self.inputs2_QParams.scale_factor.to("cpu")
        # s_x * (q_x - zero_point)

        t = torch.exp(inputs)  # exp(s_x * (q_x - zero_point))

        divisor = (self.window_size**2) * self.nhead
        if self.quant_bits <= 4:
            quant_factor = 3
        else:
            quant_factor = 2
        s_t, self.z_t, q_t = self._compute_scale_zero_point(
            self.quant_bits * quant_factor, divisor, t
        )

        # for denominator
        self.Qinput2QDenominator_LUT_dict = {}
        for i in range(len(q_t)):
            self.Qinput2QDenominator_LUT_dict[q_inputs[i].item()] = int(q_t[i].item())

        # for numerator
        q_numerator = t / (s_t * self.outputs_QParams.scale_factor.to("cpu"))
        q_numerator = q_numerator.to(dtype=torch.int32)
        q_numerator = q_numerator.round_().clamp(
            -(2 ** (self.quant_bits * 3 - 1)), 2 ** (self.quant_bits * 3 - 1) - 1
        )

        self.Qinput2QNumerator_LUT_dict = {}
        for i in range(len(q_numerator)):
            self.Qinput2QNumerator_LUT_dict[q_inputs[i].item()] = int(
                q_numerator[i].item()
            )

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.FloatTensor:
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"

        max_q_inputs = q_inputs.max(dim=-1, keepdim=True)[0]

        q_inputs = self.math_ops.intsub(
            q_inputs,
            max_q_inputs,
            self.inputs2_QParams.quant_bits + 1,
        )

        q_inputs = self.math_ops.intadd(
            q_inputs,
            self.inputs2_QParams.zero_point,
            self.inputs2_QParams.quant_bits,
        )

        q_numerator = q_inputs.clone()  # t / (S_soft * S_t)
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
                            - self.z_t
                        )
        q_denominator = q_denominator.sum(dim=-1, keepdim=True)  # sum(Q_t - Z_t)

        tmp = q_numerator / q_denominator
        tmp = (
            tmp.round_()
            .clamp(-(2 ** (self.quant_bits)), 2 ** (self.quant_bits) - 1)
            .to(dtype=torch.int32)
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        dq_outputs = self.outputs_QParams.dequantize(
            q_outputs
        )  # dequantize for visualization
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
            self.inputs2_QParams.update_quant_params(
                inputs
            )  # QParams after doing subtraction

        inputs = SimQuant.apply(inputs.to(self.device), self.inputs2_QParams)
        outputs = torch.softmax(inputs, dim=-1)

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)
        outputs = SimQuant.apply(outputs.to(self.device), self.outputs_QParams)
        return outputs
