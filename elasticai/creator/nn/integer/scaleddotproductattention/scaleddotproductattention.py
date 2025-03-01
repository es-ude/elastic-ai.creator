from pathlib import Path

import numpy as np
import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.matrixmulti import MatrixMulti
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.scaleddotproductattention.design import (
    ScaledDotProductAttention as ScaledDotProductAttentionDesign,
)
from elasticai.creator.nn.integer.softmax import SoftmaxLUT
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class ScaledDotProductAttention(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        window_size = kwargs.get("window_size")
        d_model = kwargs.get("d_model")
        nhead = kwargs.get("nhead")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.matrix_multi_score = MatrixMulti(
            name=self.name + "_matmul_score",
            x_1_dim_a=window_size,
            x_1_dim_b=nhead,
            x_1_dim_c=d_model,
            x_2_dim_a=window_size,
            x_2_dim_b=nhead,
            x_2_dim_c=d_model,
            y_dim_a=nhead,
            y_dim_b=window_size,
            y_dim_c=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            operation_mode="score",
            addtion_scale=1 / np.sqrt(d_model),
            device=device,
        )

        self.softmax = SoftmaxLUT(
            name=self.name + "_softmax",
            nhead=nhead,
            window_size=window_size,
            dim_a=nhead,
            dim_b=window_size,
            dim_c=window_size,
            quant_data_dir=self.quant_data_dir,
            quant_bits=self.quant_bits,
            device=device,
        )

        self.matrix_multi_att = MatrixMulti(
            name=self.name + "_matmul_att",
            x_1_dim_a=nhead,
            x_1_dim_b=window_size,
            x_1_dim_c=window_size,
            x_2_dim_a=window_size,
            x_2_dim_b=nhead,
            x_2_dim_c=d_model,
            y_dim_a=window_size,
            y_dim_b=nhead,
            y_dim_c=d_model,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            operation_mode="att",
            addtion_scale=None,
            device=device,
        )

        self.inputs_q_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_k_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_v_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.precomputed = False

    def create_design(self, name: str) -> ScaledDotProductAttentionDesign:
        return ScaledDotProductAttentionDesign(
            name=name,
            data_width=self.quant_bits,
            matrix_multi_score=self.matrix_multi_score,
            softmax=self.softmax,
            matrix_multi_att=self.matrix_multi_att,
            work_library_name="work",
        )

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def precompute(self) -> None:
        self.matrix_multi_score.precompute()
        self.softmax.precompute()
        self.matrix_multi_att.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_q: torch.IntTensor,
        q_k: torch.IntTensor,
        q_v: torch.IntTensor,
    ) -> torch.IntTensor:
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"

        save_quant_data(q_q, self.quant_data_dir, f"{self.name}_q_q")
        save_quant_data(q_k, self.quant_data_dir, f"{self.name}_q_k")
        save_quant_data(q_v, self.quant_data_dir, f"{self.name}_q_v")

        # MatrixMulti Score
        q_scores = self.matrix_multi_score.int_forward(
            q_inputs1=q_q,
            q_inputs2=q_k,
        )

        # Softmax
        q_att, att = self.softmax.int_forward(
            q_inputs=q_scores,
        )

        # MatrixMulti Attention
        q_context = self.matrix_multi_att.int_forward(
            q_inputs1=q_att,
            q_inputs2=q_v,
        )
        q_context = q_context.contiguous()

        save_quant_data(q_context, self.quant_data_dir, f"{self.name}_q_y")
        return q_context, att

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        given_inputs_q_QParams: object = None,
        given_inputs_k_QParams: object = None,
        given_inputs_v_QParams: object = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_q_QParams is not None:
                self.inputs_q_QParams = given_inputs_q_QParams
            else:
                self.inputs_q_QParams.update_quant_params(q)
            if given_inputs_k_QParams is not None:
                self.inputs_k_QParams = given_inputs_k_QParams
            else:
                self.inputs_k_QParams.update_quant_params(k)
            if given_inputs_v_QParams is not None:
                self.inputs_v_QParams = given_inputs_v_QParams
            else:
                self.inputs_v_QParams.update_quant_params(v)

        scores = self.matrix_multi_score.forward(
            inputs1=q,
            inputs2=k,
            given_inputs1_QParams=self.inputs_q_QParams,
            given_inputs2_QParams=self.inputs_k_QParams,
        )
        att = self.softmax.forward(
            inputs=scores, given_inputs_QParams=self.matrix_multi_score.outputs_QParams
        )
        context = self.matrix_multi_att.forward(
            inputs1=att,
            inputs2=v,
            given_inputs1_QParams=self.softmax.outputs_QParams,
            given_inputs2_QParams=self.inputs_v_QParams,
        )
        context = context.contiguous()
        self.outputs_QParams = self.matrix_multi_att.outputs_QParams
        return context, att
