import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.lstmcell import LSTMCell
from elasticai.creator.nn.integer.lstmlayer.design import LSTMLayer as LSTMLayerDesign
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams


class LSTMLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")

        self.name = kwargs.get("name")
        quant_bits = kwargs.get("quant_bits")
        self.quant_data_file_dir = Path(kwargs.get("quant_data_file_dir"))
        device = kwargs.get("device")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lstm_cell = LSTMCell(
            name=f"{self.name}_lstm_cell",
            inputs_size=inputs_size,
            hidden_size=self.hidden_size,
            quant_bits=quant_bits,
            quant_data_file_dir=self.quant_data_file_dir,
            device=device,
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_prev_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_next_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_prev_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def precompute(self):
        self.lstm_cell.precompute()
        self.precomputed = True

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
        q_h_prev: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        batch_size = q_inputs.size(0)
        seq_len = q_inputs.size(1)

        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x")
        self._save_quant_data(
            q_h_prev, self.quant_data_file_dir, f"{self.name}_q_h_prev"
        )

        q_c_prev = torch.zeros(batch_size, self.hidden_size, dtype=torch.int32).to(
            q_inputs.device
        )
        for t in range(seq_len):
            q_h_next, q_c_next = self.lstm_cell.int_forward(
                q_inputs=q_inputs[:, t, :], q_h_prev=q_h_prev, q_c_prev=q_c_prev
            )
            q_h_prev, q_c_prev = q_h_next, q_c_next

        self._save_quant_data(
            q_h_next, self.quant_data_file_dir, f"{self.name}_q_h_next"
        )
        return q_h_next

    def forward(
        self,
        inputs: torch.FloatTensor,
        h_prev: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
        given_h_prev_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

            if given_h_prev_QParams is None:
                self.h_prev_QParams.update_quant_params(h_prev)
            else:
                self.h_prev_QParams = given_h_prev_QParams

        c_prev = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        given_inputs_QParams = self.inputs_QParams
        given_h_prev_QParams = self.h_prev_QParams
        given_c_prev_QParams = self.c_prev_QParams
        for t in range(seq_len):
            h_next, c_next = self.lstm_cell.forward(
                inputs=inputs[:, t, :],
                h_prev=h_prev,
                c_prev=c_prev,
                given_inputs_QParams=given_inputs_QParams,
                given_h_prev_QParams=given_h_prev_QParams,
                given_c_prev_QParams=given_c_prev_QParams,
            )
            h_prev, c_prev = h_next, c_next

            given_c_prev_QParams = self.lstm_cell.c_next_QParams
            given_h_prev_QParams = self.lstm_cell.h_next_QParams

        self.h_next_QParams = self.lstm_cell.h_next_QParams

        return h_next
