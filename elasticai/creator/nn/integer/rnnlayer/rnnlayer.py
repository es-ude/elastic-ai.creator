import torch
import torch.nn as nn

from elasticai.creator.nn.integer.grucell import GRUCell
from elasticai.creator.nn.integer.lstmcell import LSTMCell
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
)
from elasticai.creator.nn.integer.rnnlayer.design import RNNLayer as RNNLayerDesign
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class RNNLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.cell_type = kwargs.get("cell_type")
        self.inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.batch_size = kwargs.get("batch_size")
        self.window_size = kwargs.get("window_size")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")

        if self.cell_type == "lstm":
            self.rnn_cell = LSTMCell(
                name=f"{self.name}_lstm_cell",
                inputs_size=self.inputs_size,
                hidden_size=self.hidden_size,
                window_size=self.window_size,
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
        elif self.cell_type == "gru":
            self.rnn_cell = GRUCell(
                name=f"{self.name}_gru_cell",
                inputs_size=self.inputs_size,
                hidden_size=self.hidden_size,
                window_size=self.window_size,
                quant_bits=self.quant_bits,
                quant_data_dir=self.quant_data_dir,
                device=device,
            )
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
            inputs_size=self.inputs_size,
            hidden_size=self.hidden_size,
            window_size=self.window_size,
            cell_type=self.cell_type,
            data_width=self.inputs_QParams.quant_bits,
            rnn_cell=self.rnn_cell,
            work_library_name="work",
        )

    def precompute(self):
        self.rnn_cell.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
        q_h_prev: torch.IntTensor,
        q_c_prev: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x_1")
        save_quant_data(q_h_prev, self.quant_data_dir, f"{self.name}_q_x_2")
        if q_c_prev is not None:  # to be compatible with GRU
            save_quant_data(q_c_prev, self.quant_data_dir, f"{self.name}_q_x_3")

        q_outputs = torch.zeros(
            self.batch_size, self.window_size, self.hidden_size, dtype=torch.int32
        ).to("cpu")

        for t in range(self.window_size):
            q_h_next, q_c_next = self.rnn_cell.int_forward(
                q_inputs=q_inputs[:, t, :], q_h_prev=q_h_prev, q_c_prev=q_c_prev
            )

            q_outputs[:, t, :] = q_h_next
            q_h_prev = q_h_next
            q_c_prev = q_c_next

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y_1")
        save_quant_data(q_h_next, self.quant_data_dir, f"{self.name}_q_y_2")
        if q_c_next is not None:
            save_quant_data(q_c_next, self.quant_data_dir, f"{self.name}_q_y_3")

        return q_outputs, q_h_next, q_c_next

    def forward(
        self,
        inputs: torch.FloatTensor,
        h_prev: torch.FloatTensor,
        c_prev: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module,
        given_h_prev_QParams: torch.nn.Module,
        given_c_prev_QParams: torch.nn.Module,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)
            if self.training:
                if given_h_prev_QParams is not None:
                    self.h_prev_QParams = given_h_prev_QParams
                else:
                    self.h_prev_QParams.update_quant_params(h_prev)
            if self.training:
                if given_c_prev_QParams is not None:
                    self.c_prev_QParams = given_c_prev_QParams
                else:
                    self.c_prev_QParams.update_quant_params(c_prev)

        inputs = SimQuant.apply(inputs, self.inputs_QParams)

        outputs = torch.zeros(
            self.batch_size, self.window_size, self.hidden_size, dtype=torch.float32
        ).to(inputs.device)

        for t in range(self.window_size):
            h_prev = SimQuant.apply(h_prev, self.h_prev_QParams)
            if c_prev is not None:  # to be compatible with GRU
                c_prev = SimQuant.apply(c_prev, self.c_prev_QParams)

            h_next, c_next = self.rnn_cell.forward(
                inputs=inputs[:, t, :],
                h_prev=h_prev,
                c_prev=c_prev,
                given_inputs_QParams=self.inputs_QParams,
            )

            outputs[:, t, :] = h_next.clone()
            h_prev = h_next.clone()
            c_prev = c_next.clone()

        if self.training:
            self.outputs_QParams.update_quant_params(outputs)
        outputs = SimQuant.apply(outputs, self.outputs_QParams)

        self.h_next_QParams = self.rnn_cell.h_next_QParams
        self.c_next_QParams = self.rnn_cell.c_next_QParams

        return outputs, h_next, c_next
