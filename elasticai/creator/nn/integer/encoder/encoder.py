import logging

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.encoderlayer import EncoderLayer
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.sequential import Sequential


class Encoder(nn.Module):
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
                    name=f"encoderlayer_{i}",
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

        self.sequential = Sequential(
            *self.encoder_layers,
            name=self.name,
            quant_data_file_dir=self.quant_data_file_dir,
        )

        self.precomputed = False

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        q_outputs = self.sequential.int_forward(q_inputs)
        return q_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params

        outputs = self.sequential.forward(
            inputs, given_inputs_QParams=self.inputs_QParams
        )
        self.outputs_QParams = self.sequential.submodules[-1].outputs_QParams
        return outputs
