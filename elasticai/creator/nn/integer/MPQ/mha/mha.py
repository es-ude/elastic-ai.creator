import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.MPQ.linear import Linear
from elasticai.creator.nn.integer.MPQ.mha.design import MHA as MHADesign
from elasticai.creator.nn.integer.MPQ.scaleddotproductattention import (
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
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        self.device = kwargs.get("device")

        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)
        self.MPQ_strategy = kwargs.get("MPQ_strategy")

        self.q_linear = Linear(
            name=self.name + "_q_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            bias=True,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.k_linear = Linear(
            name=self.name + "_k_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            bias=True,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )
        self.v_linear = Linear(
            name=self.name + "_v_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            bias=True,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.inner_attn_module = ScaledDotProductAttention(
            name=self.name + "_inner_attn",
            nhead=self.nhead,
            window_size=window_size,
            quant_data_dir=self.quant_data_dir,
            d_model=d_model,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )
        self.output_linear = Linear(
            name=self.name + "_output_linear",
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            bias=True,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.precomputed = False

    @property
    def inputs_q_QParams(self):
        return self.q_linear.inputs_QParams

    @property
    def inputs_k_QParams(self):
        return self.k_linear.inputs_QParams

    @property
    def inputs_v_QParams(self):
        return self.v_linear.inputs_QParams

    def outputs_QParams(self):
        return self.output_linear.outputs_QParams

    def create_design(self, name: str) -> MHADesign:
        return MHADesign(
            name=name,
            data_width=self.quant_bits_per_element["inputs"],  # TODO
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
        if self.enable_error_analysis:
            save_quant_data(
                self.output_linear.outputs_QParams.dequantize(q_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )

        return q_outputs, attn_weights

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        given_inputs_QParams: object = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        B = q.shape[0]
        L = q.shape[1]
        H = self.nhead

        q = self.q_linear.forward(
            inputs=q,
            given_inputs_QParams=given_inputs_QParams,
            enable_simquant=enable_simquant,
        )
        k = self.k_linear.forward(
            inputs=k,
            given_inputs_QParams=given_inputs_QParams,
            enable_simquant=enable_simquant,
        )
        v = self.v_linear.forward(
            inputs=v,
            given_inputs_QParams=given_inputs_QParams,
            enable_simquant=enable_simquant,
        )

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
            enable_simquant=enable_simquant,
        )

        # concat heads
        context = context.view(B, L, -1)  # B,L,H,E -> B,L,D

        outputs = self.output_linear.forward(
            inputs=context,
            given_inputs_QParams=self.inner_attn_module.matrix_multi_att.outputs_QParams,
            enable_simquant=enable_simquant,
        )

        if self.enable_error_analysis:
            save_quant_data(
                outputs,
                self.quant_data_dir,
                f"{self.name}_y",
            )

        return outputs, attn_weights
