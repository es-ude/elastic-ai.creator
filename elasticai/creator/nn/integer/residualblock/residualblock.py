import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.conv1dbn import Conv1dBN
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.relu import ReLU
from elasticai.creator.nn.integer.residualblock.design import (
    ResidualBlock as ResidualBlockDesign,
)


class ResidualBlock(DesignCreatorModule, nn.Module):
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

        self.conv1dbn_1 = Conv1dBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            seq_len=seq_len,
            name=self.name + "_1_conv1dbn",
            quant_bits=quant_bits,
            device=device,
        )

        self.conv1dbn_1_relu = ReLU(
            name=self.name + "_1_conv1dbn_relu",
            quant_bits=quant_bits,
            device=device,
        )

        self.conv1dbn_2 = Conv1dBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            seq_len=seq_len,
            name=self.name + "_2_conv1dbn",
            quant_bits=quant_bits,
            device=device,
        )

        self.shortcut = nn.ModuleList()
        if in_channels != out_channels:
            self.shortcut.append(
                Conv1dBN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                    seq_len=seq_len,
                    name=self.name + "_shortcut",
                    quant_bits=quant_bits,
                    device=device,
                )
            )

        self.add = Addition(
            name=self.name + "_add", quant_bits=quant_bits, device=device
        )
        self.relu = ReLU(name=self.name + "_relu", quant_bits=quant_bits, device=device)

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.precomputed = False

    def create_design(self, name) -> ResidualBlockDesign:
        return ResidualBlockDesign(
            name=name,
            data_width=self.inputs_QParams.quant_bits,
            in_channels=self.conv1dbn_1.in_channels,
            out_channels=self.conv1dbn_1.out_channels,
            kernel_size=self.conv1dbn_1.kernel_size,
            seq_len=self.conv1dbn_1.seq_len,
            work_library_name="work",
        )

    def precompute(self) -> None:
        assert not self.training, "int_forward should be called in eval mode"

        self.conv1dbn_1.precompute()
        self.conv1dbn_2.precompute()
        if len(self.shortcut) > 0:
            for submodule in self.shortcut:
                if hasattr(submodule, "precompute"):
                    submodule.precompute()
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
            q_inputs, self.quant_data_file_dir, f"{self.name}_conv1dbn_1_q_x"
        )
        q_outputs = self.conv1dbn_1.int_forward(q_inputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.name}_conv1dbn_1_q_y"
        )

        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.name}_conv1dbn_1_relu_q_x"
        )
        q_outputs = self.conv1dbn_1_relu.int_forward(q_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.name}_conv1dbn_1_relu_q_y"
        )

        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.name}_conv1dbn_2_q_x"
        )
        q_outputs = self.conv1dbn_2.int_forward(q_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.name}_conv1dbn_2_q_y"
        )

        self._save_quant_data(
            q_residual, self.quant_data_file_dir, f"{self.name}_shortcut_q_x"
        )
        if len(self.shortcut) > 0:
            for submodule in self.shortcut:
                q_shortcut_outputs = submodule.int_forward(q_residual)
                q_residual = q_shortcut_outputs

        else:
            q_shortcut_outputs = q_residual

        self._save_quant_data(
            q_shortcut_outputs,
            self.quant_data_file_dir,
            f"{self.name}_shortcut_q_y",
        )

        self._save_quant_data(
            q_shortcut_outputs,
            self.quant_data_file_dir,
            f"{self.name}_shortcut_add_q_x",
        )
        q_add_outputs = self.add.int_forward(
            q_inputs1=q_shortcut_outputs, q_inputs2=q_outputs
        )
        self._save_quant_data(
            q_add_outputs, self.quant_data_file_dir, f"{self.name}_shortcut_add_q_y"
        )

        self._save_quant_data(
            q_add_outputs, self.quant_data_file_dir, f"{self.name}_shortcut_relu_q_x"
        )
        q_outputs = self.relu.int_forward(q_add_outputs)
        self._save_quant_data(
            q_outputs, self.quant_data_file_dir, f"{self.name}_shortcut_relu_q_y"
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

        outputs = self.conv1dbn_1.forward(
            inputs=inputs, given_inputs_QParams=self.inputs_QParams
        )
        outputs = self.conv1dbn_1_relu.forward(
            inputs=outputs, given_inputs_QParams=self.conv1dbn_1.outputs_QParams
        )

        outputs = self.conv1dbn_2.forward(
            inputs=outputs, given_inputs_QParams=self.conv1dbn_1_relu.outputs_QParams
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
            given_inputs1_QParams=(
                self.shortcut[-1].outputs_QParams
                if len(self.shortcut) > 0
                else self.inputs_QParams
            ),
            inputs2=outputs,
            given_inputs2_QParams=self.conv1dbn_2.outputs_QParams,
        )

        outputs = self.relu.forward(
            inputs=add_outputs, given_inputs_QParams=self.add.outputs_QParams
        )

        self.outputs_QParams = self.relu.outputs_QParams

        return outputs
