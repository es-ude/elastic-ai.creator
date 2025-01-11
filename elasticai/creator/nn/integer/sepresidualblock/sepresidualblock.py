import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.conv1d import Conv1d
from elasticai.creator.nn.integer.depthconv1d import DepthConv1d
from elasticai.creator.nn.integer.pointconv1dbn import PointConv1dBN
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.relu import ReLU
from elasticai.creator.nn.integer.sepresidualblock.design import (
    SeparableResidualBlock as SeparableResidualBlockDesign,
)


class SeparableResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        kernel_size = kwargs.get("kernel_size")
        seq_len = kwargs.get("seq_len")

        self.name = kwargs.get("name")
        quant_bits = kwargs.get("quant_bits")
        self.quant_data_file_dir = Path(kwargs.get("quant_data_file_dir"))
        device = kwargs.get("device")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.depthwise_conv1d_0 = DepthConv1d(
            in_channels=kwargs.get("in_channels"),
            kernel_size=kwargs.get("kernel_size"),
            seq_len=seq_len,
            padding=1,
            groups=kwargs.get("in_channels"),
            name=self.name + "_depthwise_conv1d_0",
            quant_bits=quant_bits,
            device=device,
        )
        self.pointwise_conv1dbn_0 = PointConv1dBN(
            in_channels=kwargs.get("in_channels"),
            out_channels=kwargs.get("out_channels"),
            seq_len=seq_len,
            name=self.name + "_pointwise_conv1dbn_0",
            quant_bits=quant_bits,
            device=device,
        )
        self.pointwise_conv1dbn_0_relu = ReLU(
            name=self.name + "_pointwise_conv1dbn_0_relu",
            quant_bits=quant_bits,
            device=device,
        )

        self.depthwise_conv1d_1 = DepthConv1d(
            in_channels=out_channels,
            kernel_size=kernel_size,
            seq_len=seq_len,
            padding=1,
            groups=kwargs.get("in_channels"),
            name=self.name + "_depthwise_conv1d_0",
            quant_bits=quant_bits,
            device=device,
        )
        self.pointwise_conv1dbn_1 = PointConv1dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            seq_len=seq_len,
            name=self.name + "_pointwise_conv1dbn_0",
            quant_bits=quant_bits,
            device=device,
        )

        self.shortcut = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len=seq_len,
            kernel_size=1,
            padding="same",
            name=self.name + "_shortcut_conv1d_0",
            quant_bits=quant_bits,
            device=device,
        )

        self.add = Addition(
            name=self.name + "_add",
            num_features=out_channels,
            num_dimensions=seq_len,
            quant_bits=quant_bits,
            device=device,
        )
        self.relu = ReLU(name=self.name + "_relu", quant_bits=quant_bits, device=device)

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.precomputed = False

    def create_design(self, name: str) -> SeparableResidualBlockDesign:
        return SeparableResidualBlockDesign(
            name=name,
            data_width=self.inputs_QParams.quant_bits,
            in_channels=self.depthwise_conv1d_0.in_channels,
            out_channels=self.depthwise_conv1d_0.out_channels,
            kernel_size=self.depthwise_conv1d_0.kernel_size[0],
            seq_len=self.depthwise_conv1d_0.seq_len,
            depthwise_conv1d_0=self.depthwise_conv1d_0,
            pointwise_conv1dbn_0=self.pointwise_conv1dbn_0,
            pointwise_conv1dbn_0_relu=self.pointwise_conv1dbn_0_relu,
            depthwise_conv1d_1=self.depthwise_conv1d_1,
            pointwise_conv1dbn_1=self.pointwise_conv1dbn_1,
            shortcut=self.shortcut,
            add=self.add,
            relu=self.relu,
            work_library_name="work",
        )

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"
        self.depthwise_conv1d_0.precompute()
        self.pointwise_conv1dbn_0.precompute()

        self.depthwise_conv1d_1.precompute()
        self.pointwise_conv1dbn_1.precompute()

        self.shortcut.precompute()
        self.add.precompute()

        self.precomputed = True

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x")

        q_residual = q_inputs

        self._save_quant_data(
            q_inputs, self.quant_data_file_dir, self.depthwise_conv1d_0.name + "_q_x"
        )
        q_outputs = self.depthwise_conv1d_0.int_forward(q_inputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.depthwise_conv1d_0.name + "_q_y"
        )

        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.pointwise_conv1dbn_0.name + "_q_x"
        )
        q_outputs = self.pointwise_conv1dbn_0.int_forward(q_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.pointwise_conv1dbn_0.name + "_q_y"
        )

        self._save_quant_data(
            q_outputs,
            self.quant_data_file_dir,
            self.pointwise_conv1dbn_0_relu.name + "_q_x",
        )
        q_outputs = self.pointwise_conv1dbn_0_relu.int_forward(q_outputs)
        self._save_quant_data(
            q_outputs,
            self.quant_data_file_dir,
            self.pointwise_conv1dbn_0_relu.name + "_q_y",
        )

        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.depthwise_conv1d_1.name + "_q_x"
        )
        q_outputs = self.depthwise_conv1d_1.int_forward(q_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.depthwise_conv1d_1.name + "_q_y"
        )

        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.pointwise_conv1dbn_1.name + "_q_x"
        )
        q_outputs = self.pointwise_conv1dbn_1.int_forward(q_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.pointwise_conv1dbn_1.name + "_q_y"
        )

        self._save_quant_data(
            q_residual, self.quant_data_file_dir, self.shortcut.name + "_q_x"
        )
        q_shortcut_outputs = self.shortcut.int_forward(q_residual)
        self._save_quant_data(
            q_shortcut_outputs,
            self.quant_data_file_dir,
            self.shortcut.name + "_q_y",
        )

        self._save_quant_data(
            q_shortcut_outputs,
            self.quant_data_file_dir,
            self.add.name + "_q_x_1",
        )
        self._save_quant_data(
            q_outputs,
            self.quant_data_file_dir,
            self.add.name + "_q_x_2",
        )

        q_add_outputs = self.add.int_forward(
            q_inputs1=q_shortcut_outputs, q_inputs2=q_outputs
        )
        self._save_quant_data(
            q_add_outputs, self.quant_data_file_dir, self.add.name + "_q_y"
        )

        self._save_quant_data(
            q_add_outputs,
            self.quant_data_file_dir,
            self.relu.name + "_q_x",
        )
        q_outputs = self.relu.int_forward(q_add_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, self.relu.name + "_q_y"
        )

        self._save_quant_data(q_outputs, self.quant_data_file_dir, f"{self.name}_q_y")
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
        outputs = self.depthwise_conv1d_0.forward(
            inputs=inputs, given_inputs_QParams=self.inputs_QParams
        )

        outputs = self.pointwise_conv1dbn_0.forward(
            inputs=outputs, given_inputs_QParams=self.depthwise_conv1d_0.outputs_QParams
        )

        outputs = self.pointwise_conv1dbn_0_relu.forward(
            inputs=outputs,
            given_inputs_QParams=self.pointwise_conv1dbn_0.outputs_QParams,
        )

        outputs = self.depthwise_conv1d_1.forward(
            inputs=outputs,
            given_inputs_QParams=self.pointwise_conv1dbn_0_relu.outputs_QParams,
        )

        outputs = self.pointwise_conv1dbn_1.forward(
            inputs=outputs, given_inputs_QParams=self.depthwise_conv1d_1.outputs_QParams
        )

        shortcut_outputs = self.shortcut.forward(
            inputs=residual, given_inputs_QParams=self.inputs_QParams
        )

        add_outputs = self.add.forward(
            inputs1=shortcut_outputs,
            inputs2=outputs,
            given_inputs1_QParams=self.shortcut.outputs_QParams,
            given_inputs2_QParams=self.pointwise_conv1dbn_1.outputs_QParams,
        )

        outputs = self.relu.forward(
            inputs=add_outputs, given_inputs_QParams=self.add.outputs_QParams
        )

        self.outputs_QParams = self.relu.outputs_QParams
        return outputs
