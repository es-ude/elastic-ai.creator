import logging

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.avgpooling1d.design import (
    AVGPooling1d as AVGPooling1dDesign,
)
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricSignedQParams,
    SymmetricSignedQParams,
)
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


class AVGPooling1d(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name")
        self.in_features = kwargs.get("in_features")
        self.out_features = kwargs.get("out_features")
        self.in_num_dimensions = kwargs.get("in_num_dimensions")
        self.out_num_dimensions = kwargs.get("out_num_dimensions")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = kwargs.get("device")

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(self.device)

        self.scale_factor_Math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> AVGPooling1dDesign:
        return AVGPooling1dDesign(
            name=name,
            data_width=self.quant_bits,
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

        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        q_inputs = self.scale_factor_Math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )
        tmp = torch.sum(q_inputs, dim=2, keepdim=True).to(
            torch.int32
        )  # dim=1 for Transformer

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.scale_factor_Math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        return q_outputs.squeeze(2)

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams
        inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = F.avg_pool1d(inputs, kernel_size=inputs.size(2))

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)

        outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs.squeeze(2)
