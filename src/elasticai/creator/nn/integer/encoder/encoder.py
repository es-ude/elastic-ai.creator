import torch
import torch.nn as nn

from elasticai.creator.nn.integer.encoderlayer import EncoderLayer
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.sequential import Sequential
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        window_size: int,
        d_model: int,
        ffn_dim: int,
        nhead: int,
        num_enc_layers: int,
        quant_bits: int,
        quant_data_dir: str | None = None,
        do_int_forward: bool = False,
        device: torch.device | str = torch.device("cpu"),
        enable_fused_ffn: bool = False,
        enable_error_analysis: bool = False,
    ):
        super().__init__()

        self.name = name
        self.quant_bits = quant_bits

        self.quant_data_dir = quant_data_dir
        self.do_int_forward = do_int_forward
        self.enable_error_analysis = enable_error_analysis

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    name=f"encoderlayer_{i}",
                    enable_fused_ffn=enable_fused_ffn,
                    d_model=d_model,
                    ffn_dim=ffn_dim if ffn_dim is not None else d_model * 4,
                    nhead=nhead,
                    window_size=window_size,
                    quant_bits=self.quant_bits,
                    quant_data_dir=self.quant_data_dir,
                    device=device,
                    enable_error_analysis=self.enable_error_analysis,
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
            quant_data_dir=self.quant_data_dir,
        )

        self.precomputed = False

    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor:
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
        inputs: torch.Tensor,
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
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
