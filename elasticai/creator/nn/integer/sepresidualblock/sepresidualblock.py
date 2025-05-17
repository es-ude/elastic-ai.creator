import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.depthconv1d import DepthConv1d
from elasticai.creator.nn.integer.pointconv1dbn import PointConv1dBN
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.relu import ReLU
from elasticai.creator.nn.integer.sepresidualblock.design import (
    SepResidualBlock as SepResidualBlockDesign,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class SepResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        kernel_size = kwargs.get("kernel_size")
        seq_len = kwargs.get("seq_len")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        self.depthconv1d_0 = DepthConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            seq_len=seq_len,
            padding=1,
            groups=in_channels,
            name=self.name + "_depthconv1d_0",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.pointconv1dbn_0 = PointConv1dBN(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len=seq_len,
            name=self.name + "_pointconv1dbn_0",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.pointconv1dbn_0_relu = ReLU(
            name=self.name + "_pointconv1dbn_0_relu",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.depthconv1d_1 = DepthConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            seq_len=seq_len,
            padding=1,
            groups=in_channels,
            name=self.name + "_depthconv1d_1",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.pointconv1dbn_1 = PointConv1dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            seq_len=seq_len,
            name=self.name + "_pointconv1dbn_1",
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )

        self.shortcut = nn.ModuleList()
        if in_channels != out_channels:
            self.shortcut.append(
                DepthConv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    seq_len=seq_len,
                    padding=1,
                    groups=in_channels,
                    name=self.name + "_shortcut_depthconv1d_0",
                    quant_bits=self.quant_bits,
                    quant_data_dir=self.quant_data_dir,
                    device=device,
                )
            )
            self.shortcut.append(
                PointConv1dBN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    seq_len=seq_len,
                    name=self.name + "_shortcut_pointconv1dbn_0",
                    quant_bits=self.quant_bits,
                    quant_data_dir=self.quant_data_dir,
                    device=device,
                )
            )

        self.add = Addition(
            name=self.name + "_add",
            num_features=seq_len,
            num_dimensions=out_channels,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
        )
        self.relu = ReLU(
            name=self.name + "_relu",
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

    def create_design(self, name: str) -> SepResidualBlockDesign:
        return SepResidualBlockDesign(
            name=name,
            data_width=self.inputs_QParams.quant_bits,
            depthconv1d_0=self.depthconv1d_0,
            pointconv1dbn_0=self.pointconv1dbn_0,
            pointconv1dbn_0_relu=self.pointconv1dbn_0_relu,
            depthconv1d_1=self.depthconv1d_1,
            pointconv1dbn_1=self.pointconv1dbn_1,
            shortcut=self.shortcut,
            add=self.add,
            relu=self.relu,
            work_library_name="work",
        )

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"
        self.depthconv1d_0.precompute()
        self.pointconv1dbn_0.precompute()

        self.depthconv1d_1.precompute()
        self.pointconv1dbn_1.precompute()

        if len(self.shortcut) > 0:
            for submodule in self.shortcut:
                if hasattr(submodule, "precompute"):
                    submodule.precompute()
        self.add.precompute()

        self.precomputed = True

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_residual = q_inputs

        q_outputs = self.depthconv1d_0.int_forward(q_inputs)
        q_outputs = self.pointconv1dbn_0.int_forward(q_outputs)
        q_outputs = self.pointconv1dbn_0_relu.int_forward(q_outputs)

        q_outputs = self.depthconv1d_1.int_forward(q_outputs)
        q_outputs = self.pointconv1dbn_1.int_forward(q_outputs)

        if len(self.shortcut) > 0:
            for submodule in self.shortcut:
                q_shortcut_outputs = submodule.int_forward(q_residual)
                q_residual = q_shortcut_outputs
        else:
            q_shortcut_outputs = q_residual

        q_add_outputs = self.add.int_forward(
            q_inputs1=q_shortcut_outputs, q_inputs2=q_outputs
        )
        q_outputs = self.relu.int_forward(q_add_outputs)

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

        residual = inputs
        outputs = self.depthconv1d_0.forward(
            inputs=inputs, given_inputs_QParams=self.inputs_QParams
        )

        outputs = self.pointconv1dbn_0.forward(
            inputs=outputs, given_inputs_QParams=self.depthconv1d_0.outputs_QParams
        )

        outputs = self.pointconv1dbn_0_relu.forward(
            inputs=outputs,
            given_inputs_QParams=self.pointconv1dbn_0.outputs_QParams,
        )

        outputs = self.depthconv1d_1.forward(
            inputs=outputs,
            given_inputs_QParams=self.pointconv1dbn_0_relu.outputs_QParams,
        )

        outputs = self.pointconv1dbn_1.forward(
            inputs=outputs, given_inputs_QParams=self.depthconv1d_1.outputs_QParams
        )

        shortcut_given_inputs_QParams = self.inputs_QParams
        shortcut_inputs = residual
        if len(self.shortcut) > 0:
            for submodule in self.shortcut:
                shortcut_outputs = submodule.forward(
                    inputs=shortcut_inputs,
                    given_inputs_QParams=shortcut_given_inputs_QParams,
                )
                shortcut_inputs = shortcut_outputs
                shortcut_given_inputs_QParams = submodule.outputs_QParams
        else:
            shortcut_outputs = shortcut_inputs

        add_outputs = self.add.forward(
            inputs1=shortcut_outputs,
            inputs2=outputs,
            given_inputs1_QParams=(
                self.shortcut[-1].outputs_QParams
                if len(self.shortcut) > 0
                else self.inputs_QParams
            ),
            given_inputs2_QParams=self.pointconv1dbn_1.outputs_QParams,
        )

        outputs = self.relu.forward(
            inputs=add_outputs, given_inputs_QParams=self.add.outputs_QParams
        )

        self.outputs_QParams = self.relu.outputs_QParams
        return outputs
