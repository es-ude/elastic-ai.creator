import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.encoderlayer.design import (
    EncoderLayer as EncoderLayerDesign,
)
from elasticai.creator.nn.integer.ffn import FeedForwardNetwork
from elasticai.creator.nn.integer.mha import MultiHeadAttention
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class EncoderLayer(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        d_model = kwargs.get("d_model")
        window_size = kwargs.get("window_size")
        nhead = kwargs.get("nhead")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.mha = MultiHeadAttention(
            name=self.name + "_mha",
            nhead=nhead,
            d_model=d_model,
            window_size=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.mha_add = Addition(
            name=self.name + "_mha_add",
            num_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.mha_norm = BatchNorm1d(
            name=self.name + "_mha_norm",
            norm_dim=d_model,
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.ffn = FeedForwardNetwork(
            name=self.name + "_ffn",
            num_dimensions=window_size,
            d_model=d_model,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.ffn_add = Addition(
            name=self.name + "_ffn_add",
            num_features=d_model,
            num_dimensions=window_size,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.ffn_norm = BatchNorm1d(
            name=self.name + "_ffn_norm",
            norm_dim=d_model,
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.precomputed = False

    def create_design(self, name: str) -> EncoderLayerDesign:
        return EncoderLayerDesign(
            name=name,
            data_width=self.quant_bits,
            mha=self.mha,
            mha_add=self.mha_add,
            mha_norm=self.mha_norm,
            ffn=self.ffn,
            ffn_add=self.ffn_add,
            ffn_norm=self.ffn_norm,
            work_library_name="work",
        )

    def precompute(self) -> None:
        self.mha.precompute()
        self.mha_add.precompute()
        self.mha_norm.precompute()
        self.ffn.precompute()
        self.ffn_add.precompute()
        self.ffn_norm.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        # MHA + ADD + NORM
        q_mha_outputs, mha_attns = self.mha.int_forward(
            q_q=q_inputs,
            q_k=q_inputs,
            q_v=q_inputs,
        )
        q_mha_add_outputs = self.mha_add.int_forward(
            q_inputs1=q_inputs,
            q_inputs2=q_mha_outputs,
        )
        q_mha_norm_outputs = self.mha_norm.int_forward(
            q_inputs=q_mha_add_outputs,
        )

        # FFN + ADD + NORM
        q_ffn_outputs = self.ffn.int_forward(
            q_inputs=q_mha_norm_outputs,
        )
        q_ffn_add_outputs = self.ffn_add.int_forward(
            q_inputs1=q_mha_norm_outputs,
            q_inputs2=q_ffn_outputs,
        )
        q_ffn_norm_outputs = self.ffn_norm.int_forward(
            q_inputs=q_ffn_add_outputs,
        )

        save_quant_data(q_ffn_norm_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_ffn_norm_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: object = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)

        mha_outputs, mha_attns = self.mha.forward(
            q=inputs, k=inputs, v=inputs, given_inputs_QParams=self.inputs_QParams
        )

        mha_add_outputs = self.mha_add.forward(
            inputs1=inputs,
            inputs2=mha_outputs,
            given_inputs1_QParams=self.inputs_QParams,
            given_inputs2_QParams=self.mha.outputs_QParams,
        )

        mha_norm_outputs = self.mha_norm.forward(
            inputs=mha_add_outputs,
            given_inputs_QParams=self.mha_add.outputs_QParams,
        )

        ffn_outputs = self.ffn.forward(
            inputs=mha_norm_outputs,
            given_inputs_QParams=self.mha_norm.outputs_QParams,
        )
        ffn_add_outputs = self.ffn_add.forward(
            inputs1=mha_norm_outputs,
            inputs2=ffn_outputs,
            given_inputs1_QParams=self.mha_norm.outputs_QParams,
            given_inputs2_QParams=self.ffn.outputs_QParams,
        )
        ffn_norm_outputs = self.ffn_norm.forward(
            inputs=ffn_add_outputs,
            given_inputs_QParams=self.ffn_add.outputs_QParams,
        )
        self.outputs_QParams = self.ffn_norm.outputs_QParams

        return ffn_norm_outputs
