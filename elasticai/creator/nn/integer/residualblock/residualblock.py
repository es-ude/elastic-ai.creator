import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.conv1dbn import Conv1dBN
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.relu import ReLU
from elasticai.creator.nn.integer.sequential import Sequential


class ResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        kernel_size = kwargs.get("kernel_size")
        seq_len = kwargs.get("seq_len")

        self.name = kwargs.get("name")
        quant_bits = kwargs.get("quant_bits")
        quant_data_file_dir = Path(kwargs.get("quant_data_file_dir"))
        device = kwargs.get("device")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.layers = nn.ModuleList()

        self.layers.append(
            Conv1dBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                seq_len=seq_len,
                name=self.name + "_1_conv1dbn",
                quant_bits=quant_bits,
                device=device,
            )
        )
        self.layers.append(
            ReLU(
                name=self.name + "_1_conv1dbn_relu",
                quant_bits=quant_bits,
                device=device,
            )
        )

        self.layers.append(
            Conv1dBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                seq_len=seq_len,
                name=self.name + "_2_conv1dbn",
                quant_bits=quant_bits,
                device=device,
            )
        )

        self.sequential = Sequential(
            *self.layers, name=self.name, quant_data_file_dir=quant_data_file_dir
        )

        self.shortcut_layers = nn.ModuleList()
        if in_channels != out_channels:
            self.shortcut_layers.append(
                Conv1dBN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                    seq_len=seq_len,
                    name=self.name + "_shortcut_conv1dbn",
                    quant_bits=quant_bits,
                    device=device,
                )
            )
        self.shortcut_sequential = Sequential(
            *self.shortcut_layers,
            name=self.name + "_shortcut",
            quant_data_file_dir=quant_data_file_dir
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

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"

        self.sequential.precompute()
        self.shortcut_sequential.precompute()
        self.add.precompute()

        residual = q_inputs
        q_outputs = self.sequential.int_forward(q_inputs)
        q_shortcut_outputs = self.shortcut_sequential.int_forward(residual)
        q_outputs = self.add.int_forward(
            q_inputs1=q_shortcut_outputs, q_inputs2=q_outputs
        )
        q_outputs = self.relu.int_forward(q_inputs=q_outputs)

        return q_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        self.inputs_QParams = given_inputs_QParams

        residual = inputs
        outputs = self.sequential.forward(inputs)

        shortcut_outputs = self.shortcut_sequential.forward(residual)
        outputs = self.add.forward(inputs1=shortcut_outputs, inputs2=outputs)
        outputs = self.relu.forward(inputs=outputs)
        self.outputs_QParams = self.relu.outputs_QParams

        return outputs
