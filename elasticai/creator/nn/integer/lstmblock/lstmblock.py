import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.lstmblock.design import LSTMBlock as LSTMBlockDesign
from elasticai.creator.nn.integer.lstmlayer import LSTMLayer
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams


class LSTMBlock(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        inputs_size = kwargs.get("inputs_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.num_layers = kwargs.get("num_layers")
        seq_len = kwargs.get("seq_len")
        self.batch_size = kwargs.get("batch_size")

        self.name = kwargs.get("name")
        quant_bits = kwargs.get("quant_bits")
        self.quant_data_file_dir = Path(kwargs.get("quant_data_file_dir"))
        device = kwargs.get("device")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lstm_layers = nn.ModuleList(
            [
                LSTMLayer(
                    inputs_size=inputs_size,
                    hidden_size=self.hidden_size,  # TODO: check if this is correct
                    quant_bits=quant_bits,
                    seq_len=seq_len,
                    batch_size=self.batch_size,
                    name=self.name + f"lstm_layer_{i}",
                    quant_data_file_dir=self.quant_data_file_dir,
                    device=device,
                )
                for i in range(self.num_layers)
            ]
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name) -> LSTMBlockDesign:
        pass

    def precompute(self) -> None:
        for layer in self.lstm_layers:
            layer.precompute()
        self.precomputed = True

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"

        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x")
        q_h_prev = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.int32).to(
            "cpu"
        )
        q_h_prev = self.q_h_prev
        for layer in self.lstm_layers:
            q_h_next = layer.int_forward(q_inputs=q_inputs, q_h_prev=q_h_prev)
            q_h_prev = q_h_next

        self._save_quant_data(
            q_h_next, self.quant_data_file_dir, f"{self.name}_q_h_next"
        )
        return q_h_next

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is None:
                self.inputs_QParams.update_quant_params(inputs)
            else:
                self.inputs_QParams = given_inputs_QParams

        # h_prev = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float32).to(
        #     inputs.device
        # )
        h_prev = torch.randn(self.batch_size, self.hidden_size).to(inputs.device) * 1e-5

        given_h_prev_QParams = None
        for layer in self.lstm_layers:
            h_next = layer.forward(
                inputs=inputs,
                h_prev=h_prev,
                given_inputs_QParams=self.inputs_QParams,
                given_h_prev_QParams=given_h_prev_QParams,
            )
            h_prev = h_next
            print("layer.outputs_QParams: ", layer.outputs_QParams)
            given_h_prev_QParams = layer.outputs_QParams

        print(
            "self.lstm_layers[-1].outputs_QParams: ",
            self.lstm_layers[-1].outputs_QParams,
        )
        self.outputs_QParams = self.lstm_layers[-1].outputs_QParams
        return h_next
