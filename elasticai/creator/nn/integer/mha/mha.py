import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.mha.design import MHA as MHADesign
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.scaleddotproductattention import (
    ScaledDotProductAttention,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class MultiHeadAttention(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.nhead = kwargs.get("nhead")
        d_model = kwargs.get("d_model")
        window_size = kwargs.get("window_size")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.q_linear = Linear(
            name=self.name + "_q_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
            bias=True,
        )

        self.k_linear = Linear(
            name=self.name + "_k_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
            bias=True,
        )
        self.v_linear = Linear(
            name=self.name + "_v_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
            bias=True,
        )

        self.inner_attn_module = ScaledDotProductAttention(
            name=self.name + "_inner_attn",
            nhead=self.nhead,
            window_size=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            d_model=d_model,
        )
        self.output_linear = Linear(
            name=self.name + "_output_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
            bias=True,
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name: str) -> MHADesign:
        return MHADesign(
            name=name,
            data_width=self.quant_bits,
            q_linear=self.q_linear,
            k_linear=self.k_linear,
            v_linear=self.v_linear,
            inner_attn_module=self.inner_attn_module,
            output_linear=self.output_linear,
            work_library_name="work",
        )

    def precompute(self) -> None:
        self.q_linear.precompute()
        self.k_linear.precompute()
        self.v_linear.precompute()
        self.inner_attn_module.precompute()
        self.output_linear.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_q: torch.IntTensor,
        q_k: torch.IntTensor,
        q_v: torch.IntTensor,
    ):
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"

        save_quant_data(q_q, self.quant_data_dir, f"{self.name}_q_q")
        save_quant_data(q_k, self.quant_data_dir, f"{self.name}_q_k")
        save_quant_data(q_v, self.quant_data_dir, f"{self.name}_q_v")

        B = q_q.shape[0]
        L = q_q.shape[1]
        H = self.nhead

        q_linear_outputs = self.q_linear.int_forward(
            q_inputs=q_q,
        )
        k_linear_outputs = self.k_linear.int_forward(
            q_inputs=q_k,
        )
        v_linear_outputs = self.v_linear.int_forward(
            q_inputs=q_v,
        )

        q_linear_outputs = q_linear_outputs.view(B, L, H, -1)
        k_linear_outputs = k_linear_outputs.view(B, L, H, -1)
        v_linear_outputs = v_linear_outputs.view(B, L, H, -1)

        # execute scaled dot product attention
        q_context, attn_weights = self.inner_attn_module.int_forward(
            q_q=q_linear_outputs,
            q_k=k_linear_outputs,
            q_v=v_linear_outputs,
        )
        q_context = q_context.view(B, L, -1)  # B,L,H,E -> B,L,D

        # Linear outputs
        q_outputs = self.output_linear.int_forward(
            q_inputs=q_context,
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")

        return q_outputs, attn_weights

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        given_inputs_QParams: object = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(q)

        B = q.shape[0]
        L = q.shape[1]
        H = self.nhead

        q = self.q_linear.forward(inputs=q, given_inputs_QParams=self.inputs_QParams)
        k = self.k_linear.forward(inputs=k, given_inputs_QParams=self.inputs_QParams)
        v = self.v_linear.forward(inputs=v, given_inputs_QParams=self.inputs_QParams)

        q = q.view(B, L, H, -1)  # B,L,E -> B,L,H,E/H
        k = k.view(B, L, H, -1)
        v = v.view(B, L, H, -1)

        context, attn_weights = self.inner_attn_module.forward(
            q=q,
            k=k,
            v=v,
            given_inputs_q_QParams=self.q_linear.outputs_QParams,
            given_inputs_k_QParams=self.k_linear.outputs_QParams,
            given_inputs_v_QParams=self.v_linear.outputs_QParams,
        )

        # concat heads
        context = context.view(B, L, -1)  # B,L,H,E -> B,L,D

        outputs = self.output_linear.forward(
            inputs=context,
            given_inputs_QParams=self.inner_attn_module.matrix_multi_att.outputs_QParams,
        )

        self.outputs_QParams = self.output_linear.outputs_QParams

        return outputs, attn_weights
