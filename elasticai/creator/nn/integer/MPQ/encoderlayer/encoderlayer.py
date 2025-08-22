import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.MPQ.addition import Addition
from elasticai.creator.nn.integer.MPQ.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.MPQ.encoderlayer.design import (
    EncoderLayer as EncoderLayerDesign,
)
from elasticai.creator.nn.integer.MPQ.ffn import FeedForwardNetwork
from elasticai.creator.nn.integer.MPQ.mha import MultiHeadAttention
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class EncoderLayer(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        d_model = kwargs.get("d_model")
        ffn_dim = kwargs.get("ffn_dim")
        window_size = kwargs.get("window_size")
        nhead = kwargs.get("nhead")

        self.name = kwargs.get("name")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        self.device = kwargs.get("device")

        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)
        self.MPQ_strategy = kwargs.get("MPQ_strategy")

        mha_name = self.name + "_mha"
        ffn_name = self.name + "_ffn"

        self.mha = MultiHeadAttention(
            name=mha_name,
            nhead=nhead,
            d_model=d_model,
            window_size=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )
        self.mha_add = Addition(
            name=mha_name + "_add",
            num_features=d_model,
            num_dimensions=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.mha_norm = BatchNorm1d(
            name=mha_name + "_norm",
            norm_dim=d_model,
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.ffn = FeedForwardNetwork(
            name=ffn_name,
            num_dimensions=window_size,
            d_model=d_model,
            ffn_dim=ffn_dim,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.ffn_add = Addition(
            name=ffn_name + "_add",
            num_features=d_model,
            num_dimensions=window_size,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.ffn_norm = BatchNorm1d(
            name=ffn_name + "_norm",
            norm_dim=d_model,
            in_features=d_model,
            out_features=d_model,
            num_dimensions=window_size,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )
        self.precomputed = False

    @property
    def inputs_QParams(self):
        return self.mha.inputs_q_QParams

    @property
    def outputs_QParams(self):
        return self.ffn.outputs_QParams

    def create_design(self, name: str) -> EncoderLayerDesign:
        return EncoderLayerDesign(
            name=name,
            data_width=self.mha.q_linear.quant_bits_per_element["inputs"],  # TODO
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
        if self.enable_error_analysis:
            save_quant_data(
                self.ffn_norm.outputs_QParams.dequantize(q_ffn_norm_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_ffn_norm_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: object = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        mha_outputs, mha_attns = self.mha.forward(
            q=inputs,
            k=inputs,
            v=inputs,
            given_inputs_QParams=given_inputs_QParams,
            enable_simquant=enable_simquant,
        )

        mha_add_outputs = self.mha_add.forward(
            inputs1=inputs,
            inputs2=mha_outputs,
            given_inputs1_QParams=given_inputs_QParams,
            given_inputs2_QParams=self.mha.output_linear.outputs_QParams,
            enable_simquant=enable_simquant,
        )

        mha_norm_outputs = self.mha_norm.forward(
            inputs=mha_add_outputs,
            given_inputs_QParams=self.mha_add.outputs_QParams,
            enable_simquant=enable_simquant,
        )

        ffn_outputs = self.ffn.forward(
            inputs=mha_norm_outputs,
            given_inputs_QParams=self.mha_norm.outputs_QParams,
            enable_simquant=enable_simquant,
        )
        ffn_add_outputs = self.ffn_add.forward(
            inputs1=mha_norm_outputs,
            inputs2=ffn_outputs,
            given_inputs1_QParams=self.mha_norm.outputs_QParams,
            given_inputs2_QParams=self.ffn.fc2.outputs_QParams,
            enable_simquant=enable_simquant,
        )
        ffn_norm_outputs = self.ffn_norm.forward(
            inputs=ffn_add_outputs,
            given_inputs_QParams=self.ffn_add.outputs_QParams,
            enable_simquant=enable_simquant,
        )

        if self.enable_error_analysis:
            save_quant_data(
                ffn_norm_outputs,
                self.quant_data_dir,
                f"{self.name}_y",
            )

        return ffn_norm_outputs
