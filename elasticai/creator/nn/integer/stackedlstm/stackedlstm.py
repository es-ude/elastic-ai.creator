import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.lstmlayer import LSTMLayer
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.stackedlstm.design import (
    StackedLSTM as StackedLSTMDesign,
)


class StackedLSTM(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.num_layers = kwargs.get("num_layers")
        self.seq_len = kwargs.get("seq_len")
        self.batch_size = kwargs.get("batch_size")

        device = kwargs.get("device")
        self.name = kwargs.get("name")
        quant_bits = kwargs.get("quant_bits")
        self.quant_data_file_dir = Path(kwargs.get("quant_data_file_dir"))
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_layers.append(
                LSTMLayer(
                    inputs_size=self.inputs_size if i == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                    quant_bits=quant_bits,
                    seq_len=self.seq_len,
                    batch_size=self.batch_size,
                    name=self.name + f"lstm_layer_{i}",
                    quant_data_file_dir=self.quant_data_file_dir,
                    device=device,
                )
            )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_prev_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_prev_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name) -> StackedLSTMDesign:
        return StackedLSTMDesign(
            name=name,
            data_width=self.inputs_QParams.quant_bits,
            lstm_layers=self.lstm_layers,
            num_layers=self.num_layers,
            work_library_name="work",
        )

    def precompute(self) -> None:
        for layer in self.lstm_layers:
            layer.precompute()

        h_0 = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32)
        c_0 = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32)

        self.q_h_0 = self.h_prev_QParams.quantize(h_0)
        self.q_c_0 = self.c_prev_QParams.quantize(c_0)

        self.precomputed = True

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x")

        q_h_prev = self.q_h_0
        q_c_prev = self.q_c_0
        for layer in self.lstm_layers:
            q_outputs, q_h_next, q_c_next = layer.int_forward(
                q_inputs=q_inputs,
                q_h_prev=q_h_prev,
                q_c_prev=q_c_prev,
            )
            q_inputs = q_outputs
            q_h_prev = q_h_next
            q_c_prev = q_c_next

        self._save_quant_data(q_h_next, self.quant_data_file_dir, f"{self.name}_q_y")
        return q_h_next

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)

        h_0 = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
            inputs.device
        )
        c_0 = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
            inputs.device
        )
        if self.training:
            self.h_prev_QParams.update_quant_params(h_0)
            self.c_prev_QParams.update_quant_params(c_0)

        h_prev = h_0
        c_prev = c_0
        given_inputs_QParams = self.inputs_QParams
        given_h_prev_QParams = self.h_prev_QParams
        given_c_prev_QParams = self.c_prev_QParams
        for layer in self.lstm_layers:
            outputs, h_next, c_next = layer.forward(
                inputs=inputs,
                h_prev=h_prev,
                c_prev=c_prev,
                given_inputs_QParams=given_inputs_QParams,
                given_h_prev_QParams=given_h_prev_QParams,
                given_c_prev_QParams=given_c_prev_QParams,
            )
            inputs = outputs
            h_prev = h_next
            c_prev = c_next
            given_inputs_QParams = layer.outputs_QParams
            given_h_prev_QParams = layer.h_next_QParams
            given_c_prev_QParams = layer.c_next_QParams

        self.outputs_QParams = self.lstm_layers[-1].h_next_QParams
        return h_next
