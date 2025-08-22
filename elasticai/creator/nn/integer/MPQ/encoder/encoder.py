import logging

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.MPQ.encoderlayer import EncoderLayer
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.sequential import Sequential
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        window_size = kwargs.get("window_size")
        d_model = kwargs.get("d_model")
        ffn_dim = kwargs.get("ffn_dim")
        nhead = kwargs.get("nhead")
        num_enc_layers = kwargs.get("num_enc_layers")

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs", "outputs"]
        self.quant_bits_per_element = None

        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        self.do_int_forward = kwargs.get("do_int_forward")
        self.device = kwargs.get("device")

        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.MPQ_strategy = kwargs.get("MPQ_strategy")

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    name=f"encoderlayer_{i}",
                    d_model=d_model,
                    ffn_dim=ffn_dim if ffn_dim is not None else d_model * 4,
                    nhead=nhead,
                    window_size=window_size,
                    quant_data_dir=self.quant_data_dir,
                    device=self.device,
                    enable_error_analysis=self.enable_error_analysis,
                    MPQ_strategy=self.MPQ_strategy,
                )
                for i in range(num_enc_layers)
            ]
        )

        self.sequential = Sequential(
            *self.encoder_layers,
            name=self.name,
            quant_data_dir=self.quant_data_dir,
        )
        self.precomputed = False

    def set_quant_bits_from_config(self, quant_configs):
        quant_bits_per_element = {}
        for element in self.quantizable_elements:
            key = f"{self.name}.{element}"
            quant_bits_per_element[element] = quant_configs.get(key)
        self.quant_bits_per_element = quant_bits_per_element
        self._init_element_Qparams()

    def _init_element_Qparams(self):
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["inputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits_per_element["outputs"],
            observer=GlobalMinMaxObserver(),
        ).to(self.device)

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_outputs = self.sequential.int_forward(q_inputs)
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        if self.enable_error_analysis:
            save_quant_data(
                self.sequential[-1].outputs_QParams.dequantize(q_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is not None:
                    self.inputs_QParams = given_inputs_QParams
                else:
                    self.inputs_QParams.update_quant_params

        outputs = self.sequential.forward(
            inputs,
            given_inputs_QParams=self.inputs_QParams,
            enable_simquant=enable_simquant,
        )
        if enable_simquant:
            self.outputs_QParams = self.sequential.submodules[-1].outputs_QParams
            if self.enable_error_analysis:
                save_quant_data(
                    outputs,
                    self.quant_data_dir,
                    f"{self.name}_y",
                )
        return outputs
