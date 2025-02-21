import logging
from pathlib import Path

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.encoder.design import Encoder as EncoderDesign
from elasticai.creator.nn.integer.encoderlayer import EncoderLayer
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams


class Encoder(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        window_size = kwargs.get("window_size")
        d_model = kwargs.get("d_model")
        nhead = kwargs.get("nhead")
        num_enc_layers = kwargs.get("num_enc_layers")

        device = kwargs.get("device")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.do_int_forward = kwargs.get("do_int_forward")
        self.quant_data_file_dir = kwargs.get("quant_data_file_dir")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    name=f"enc_layer_{i}",
                    d_model=d_model,
                    nhead=nhead,
                    window_size=window_size,
                    quant_bits=self.quant_bits,
                    quant_data_file_dir=self.quant_data_file_dir,
                    device=device,
                )
                for i in range(num_enc_layers)
            ]
        )
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.precomputed = False

    def create_design(self, name: str) -> EncoderDesign:
        # TODO: support more than 1 encoder layer
        return EncoderDesign(
            name=name,
            data_width=self.quant_bits,
            encoder_layers=self.encoder_layers,
            work_library_name="work",
        )

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def precompute(self) -> None:
        for layer in self.encoder_layers:
            layer.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert not self.training, "int_forward should be called in eval mode"
        assert self.precomputed, "precompute should be called before int_forward"
        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x")

        encoder_attns = []
        current_layer_inputs = q_inputs
        for layer in self.encoder_layers:
            self._save_quant_data(
                current_layer_inputs, self.quant_data_file_dir, f"{layer.name}_q_x"
            )
            q_layer_outputs, layer_attns = layer.int_forward(
                q_inputs=current_layer_inputs,
            )
            self._save_quant_data(
                q_layer_outputs, self.quant_data_file_dir, f"{layer.name}_q_y"
            )
            current_layer_inputs = q_layer_outputs
            encoder_attns.append(layer_attns)

        q_encoder_outputs = q_layer_outputs
        self._save_quant_data(
            q_encoder_outputs, self.quant_data_file_dir, f"{self.name}_q_y"
        )
        return q_encoder_outputs, encoder_attns

    def forward(self, inputs: torch.FloatTensor, given_inputs_QParams: object = None):
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)
        encoder_attns = []
        current_inputs_QParams = self.inputs_QParams
        current_layer_inputs = inputs

        for layer in self.encoder_layers:
            layer_outputs, layer_attns = layer.forward(
                inputs=current_layer_inputs, given_inputs_QParams=current_inputs_QParams
            )
            encoder_attns.append(layer_attns)
            current_layer_inputs = layer_outputs
            current_inputs_QParams = layer.outputs_QParams
        encoder_outputs = layer_outputs
        self.outputs_QParams = current_inputs_QParams
        return encoder_outputs, encoder_attns
