import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.mha.design import MHA as MHADesign
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.scaleddotproductattention import (
    ScaledDotProductAttention,
)


class MultiHeadAttention(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.nhead = kwargs.get("nhead")
        d_model = kwargs.get("d_model")
        window_size = kwargs.get("window_size")
        self.quant_data_file_dir = kwargs.get("quant_data_file_dir")

        self.name = kwargs.get("name")
        device = kwargs.get("device")
        self.quant_bits = kwargs.get("quant_bits")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.q_linear = (
            Linear(  # TODO: missing num_dimensions(=window_size) for VHDL templates
                name=self.name + "_q_linear",
                in_features=d_model,
                out_features=d_model,
                quant_bits=self.quant_bits,
                device=device,
                bias=True,
            )
        )

        self.k_linear = (
            Linear(  # TODO: missing num_dimensions(=window_size) for VHDL templates
                name=self.name + "_k_linear",
                in_features=d_model,
                out_features=d_model,
                quant_bits=self.quant_bits,
                device=device,
                bias=True,
            )
        )
        self.v_linear = (
            Linear(  # TODO: missing num_dimensions(=window_size) for VHDL templates
                name=self.name + "_v_linear",
                in_features=d_model,
                out_features=d_model,
                quant_bits=self.quant_bits,
                device=device,
                bias=True,
            )
        )

        self.inner_attn_module = ScaledDotProductAttention(
            name=self.name + "_inner_attn",
            quant_bits=self.quant_bits,
            nhead=self.nhead,
            window_size=kwargs.get("window_size"),
            quant_data_file_dir=self.quant_data_file_dir,
            d_model=d_model,
        )
        self.output_linear = (
            Linear(  # TODO: missing num_dimensions(=window_size) for VHDL templates
                name=self.name + "_output_linear",
                in_features=d_model,
                out_features=d_model,
                quant_bits=self.quant_bits,
                device=device,
                bias=True,
            )
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name: str) -> MHADesign:
        return MHADesign(name=name)

    def precompute(self) -> None:
        self.q_linear.precompute()
        self.k_linear.precompute()
        self.v_linear.precompute()
        self.inner_attn_module.precompute()
        self.output_linear.precompute()
        self.precomputed = True

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def int_forward(
        self,
        q_q: torch.IntTensor,
        q_k: torch.IntTensor,
        q_v: torch.IntTensor,
    ):
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"

        self._save_quant_data(q_q, self.quant_data_file_dir, f"{self.name}_q_q")
        self._save_quant_data(q_k, self.quant_data_file_dir, f"{self.name}_q_k")
        self._save_quant_data(q_v, self.quant_data_file_dir, f"{self.name}_q_v")

        B = q_q.shape[0]
        L = q_q.shape[1]
        H = self.nhead

        q_linear_outputs = self.q_linear.int_forward(
            q_inputs=q_q,
        ).view(B, L, H, -1)
        k_linear_outputs = self.k_linear.int_forward(
            q_inputs=q_k,
        ).view(B, L, H, -1)
        v_linear_outputs = self.v_linear.int_forward(
            q_inputs=q_v,
        ).view(B, L, H, -1)

        # execute scaled dot product attention
        self._save_quant_data(
            q_linear_outputs,
            self.quant_data_file_dir,
            f"{self.inner_attn_module.name}_q_q",
        )
        self._save_quant_data(
            k_linear_outputs,
            self.quant_data_file_dir,
            f"{self.inner_attn_module.name}_q_k",
        )
        self._save_quant_data(
            v_linear_outputs,
            self.quant_data_file_dir,
            f"{self.inner_attn_module.name}_q_v",
        )

        q_context, attn_weights = self.inner_attn_module.int_forward(
            q_q=q_linear_outputs,
            q_k=k_linear_outputs,
            q_v=v_linear_outputs,
        )
        q_context = q_context.view(B, L, -1)  # B,L,H,E -> B,L,D
        self._save_quant_data(
            q_context, self.quant_data_file_dir, f"{self.inner_attn_module.name}_q_y"
        )

        # Linear outputs
        self._save_quant_data(
            q_context, self.quant_data_file_dir, f"{self.output_linear.name}_q_x"
        )
        q_outputs = self.output_linear.int_forward(
            q_inputs=q_context,
        )
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.output_linear.name}_q_y"
        )

        return q_outputs, attn_weights

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        given_inputs_QParams: object,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(q)

        B = q.shape[0]
        L = q.shape[1]
        H = self.nhead

        # linear projection and split heads
        q = self.q_linear(inputs=q, given_inputs_QParams=self.inputs_QParams)
        k = self.k_linear(inputs=k, given_inputs_QParams=self.inputs_QParams)
        v = self.v_linear(inputs=v, given_inputs_QParams=self.inputs_QParams)

        q = q.view(B, L, H, -1)  # B,L,E -> B,L,H,E/H
        k = k.view(B, L, H, -1)
        v = v.view(B, L, H, -1)

        # execute scaled dot product attention
        context, attn_weights = self.inner_attn_module(
            q=q,
            k=k,
            v=v,
            given_inputs_q_QParams=self.q_linear.outputs_QParams,
            given_inputs_k_QParams=self.k_linear.outputs_QParams,
            given_inputs_v_QParams=self.v_linear.outputs_QParams,
        )

        # concat heads
        context = context.view(B, L, -1)  # B,L,H,E -> B,L,D

        outputs = self.output_linear(
            inputs=context,
            given_inputs_QParams=self.inner_attn_module.matrix_multi_att.outputs_QParams,
        )

        self.outputs_QParams = self.output_linear.outputs_QParams

        return outputs, attn_weights
