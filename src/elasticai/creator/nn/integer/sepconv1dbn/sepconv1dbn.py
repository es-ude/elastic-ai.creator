import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.sepconv1dbn import DepthConv1d, PointConv1dBN
from elasticai.creator.nn.integer.sepconv1dbn.design import (
    SepConv1dBN as SepConv1dBNDesign,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class SepConv1dBN(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        kernel_size = kwargs.get("kernel_size")
        padding = kwargs.get("padding")
        self.seq_len = kwargs.get("seq_len")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.depthconv1d = DepthConv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=kernel_size,  # 3
            padding=padding,  # 1
            seq_len=self.seq_len,
            name=self.name + "_depthconv1d_0",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.pointconv1dbn = PointConv1dBN(
            in_channels=self.in_channels,
            out_channels=out_channels,
            seq_len=self.seq_len,
            name=self.name + "_pointconv1dbn_0",
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

    def create_design(self, name: str) -> SepConv1dBNDesign:
        return SepConv1dBNDesign(
            name=name,
            data_width=self.quant_bits,
            depthconv1d=self.depthconv1d,
            pointconv1dbn=self.pointconv1dbn,
            work_library_name="work",
        )

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"
        self.depthconv1d.precompute()
        self.pointconv1dbn.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_depthconv1d_outputs = self.depthconv1d.int_forward(
            q_inputs=q_inputs,
        )
        q_pointconv1dbn_outputs = self.pointconv1dbn.int_forward(
            q_inputs=q_depthconv1d_outputs,
        )

        save_quant_data(
            q_pointconv1dbn_outputs, self.quant_data_dir, f"{self.name}_q_y"
        )
        return q_pointconv1dbn_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = given_inputs_QParams

        DepthConv1d_outputs = self.depthconv1d.forward(
            inputs=inputs,
            given_inputs_QParams=self.inputs_QParams,
            enable_simquant=enable_simquant,
        )
        PointConv1dBN_outputs = self.pointconv1dbn.forward(
            inputs=DepthConv1d_outputs,
            given_inputs_QParams=self.depthconv1d.outputs_QParams,
            enable_simquant=enable_simquant,
        )
        if enable_simquant:
            self.outputs_QParams = self.pointconv1dbn.outputs_QParams

        return PointConv1dBN_outputs
