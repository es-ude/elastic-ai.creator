import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.lstmcell import LSTMCell
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.rnnlayer.design import RNNLayer as RNNLayerDesign


class RNNLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.batch_size = kwargs.get("batch_size")
        self.seq_len = kwargs.get("seq_len")
        self.cell_type = kwargs.get("cell_type")

        device = kwargs.get("device")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_file_dir = Path(kwargs.get("quant_data_file_dir"))
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.cell_type == "lstm":
            self.rnn_cell = LSTMCell(
                name=f"{self.name}_lstm_cell",
                inputs_size=inputs_size,
                hidden_size=self.hidden_size,
                quant_bits=self.quant_bits,
                quant_data_file_dir=self.quant_data_file_dir,
                device=device,
            )
        elif self.cell_type == "gru":
            pass
        else:
            raise ValueError(f"Unsupported cell type: {self.cell_type}")

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_prev_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_prev_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.h_next_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.c_next_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name: str) -> RNNLayerDesign:
        return RNNLayerDesign(
            name=name,
            data_width=self.inputs_QParams.quant_bits,
            rnn_cell=self.rnn_cell,
            work_library_name="work",
        )

    def precompute(self):
        self.rnn_cell.precompute()

        self.precomputed = True

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
        q_h_prev: torch.IntTensor,
        q_c_prev: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x_1")
        self._save_quant_data(q_h_prev, self.quant_data_file_dir, f"{self.name}_q_x_2")
        self._save_quant_data(q_c_prev, self.quant_data_file_dir, f"{self.name}_q_x_3")

        q_outputs = torch.zeros(
            self.batch_size, self.seq_len, self.hidden_size, dtype=torch.int32
        ).to("cpu")

        for t in range(self.seq_len):
            q_h_next, q_c_next = self.rnn_cell.int_forward(
                q_inputs=q_inputs[:, t, :], q_h_prev=q_h_prev, q_c_prev=q_c_prev
            )

            q_outputs[:, t, :] = q_h_next
            q_h_prev = q_h_next
            q_c_prev = q_c_next

        self._save_quant_data(q_outputs, self.quant_data_file_dir, f"{self.name}_q_y_1")
        self._save_quant_data(q_h_next, self.quant_data_file_dir, f"{self.name}_q_y_2")
        self._save_quant_data(q_c_next, self.quant_data_file_dir, f"{self.name}_q_y_3")

        return q_outputs, q_h_next, q_c_next

    def forward(
        self,
        inputs: torch.FloatTensor,
        h_prev: torch.FloatTensor,
        c_prev: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
        given_h_prev_QParams: torch.nn.Module = None,
        given_c_prev_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)
            if given_h_prev_QParams is not None:
                self.h_prev_QParams = given_h_prev_QParams
            else:
                self.h_prev_QParams.update_quant_params(h_prev)

            if given_c_prev_QParams is not None:
                self.c_prev_QParams = given_c_prev_QParams
            else:
                self.c_prev_QParams.update_quant_params(c_prev)

        inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = torch.zeros(
            self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32
        ).to(inputs.device)

        for t in range(self.seq_len):
            h_prev = SimQuant.apply(h_prev, self.h_prev_QParams)
            c_prev = SimQuant.apply(c_prev, self.c_prev_QParams)

            h_next, c_next = self.rnn_cell.forward(
                inputs=inputs[:, t, :],
                h_prev=h_prev,
                c_prev=c_prev,
                given_inputs_QParams=self.inputs_QParams,
            )
            outputs[:, t, :] = h_next
            h_prev, c_prev = h_next, c_next

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)
        outputs = SimQuant.apply(outputs, self.outputs_QParams)

        self.h_next_QParams = self.rnn_cell.h_next_QParams
        self.c_next_QParams = self.rnn_cell.c_next_QParams

        return outputs, h_next, c_next
