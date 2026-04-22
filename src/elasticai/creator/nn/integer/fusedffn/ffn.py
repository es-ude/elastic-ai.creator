import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.fusedffn.design import FFN as FFNDesign
from elasticai.creator.nn.integer.fusedffn.fc1relu import LinearReLU
from elasticai.creator.nn.integer.fusedffn.fc2 import Linear
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class FusedFeedForwardNetwork(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        d_model = kwargs.get("d_model")
        ffn_dim = kwargs.get("ffn_dim")
        num_dimensions = kwargs.get("num_dimensions")

        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        device = kwargs.get("device")
        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.fc1relu = LinearReLU(
            name=self.name + "_fc1relu",
            in_features=d_model,
            out_features=ffn_dim,
            num_dimensions=num_dimensions,
            bias=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
            enable_error_analysis=self.enable_error_analysis,
        )

        self.fc2 = Linear(
            name=self.name + "_fc2",
            in_features=ffn_dim,
            out_features=d_model,
            num_dimensions=num_dimensions,
            bias=True,
            quant_bits=self.quant_bits,
            quant_data_dir=self.quant_data_dir,
            device=device,
            enable_error_analysis=self.enable_error_analysis,
        )

        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.precomputed = False

    def create_design(self, name: str) -> FFNDesign:
        return FFNDesign(
            name=name,
            data_width=self.quant_bits,
            fc1relu=self.fc1relu,
            fc2=self.fc2,
            work_library_name="work",
        )

    def precompute(self):
        assert not self.training, "int_forward should be called in eval mode"
        self.fc1relu.precompute()
        self.fc2.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_f1relu_outputs = self.fc1relu.int_forward(
            q_inputs=q_inputs,
        )
        q_f2_outputs = self.fc2.int_forward(
            q_inputs=q_f1relu_outputs,
        )

        save_quant_data(q_f2_outputs, self.quant_data_dir, f"{self.name}_q_y")
        if self.enable_error_analysis:
            save_quant_data(
                self.outputs_QParams.dequantize(q_f2_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
        return q_f2_outputs

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: nn.Module = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is not None:
                    self.inputs_QParams = given_inputs_QParams
                else:
                    self.inputs_QParams.update_quant_params(inputs)

        f1relu_outputs = self.fc1relu.forward(
            inputs=inputs,
            given_inputs_QParams=self.inputs_QParams,
            enable_simquant=enable_simquant,
        )

        f2_outputs = self.fc2.forward(
            inputs=f1relu_outputs,
            given_inputs_QParams=self.fc1relu.outputs_QParams,
            enable_simquant=enable_simquant,
        )
        if enable_simquant:
            self.outputs_QParams = self.fc2.outputs_QParams
            if self.enable_error_analysis:
                save_quant_data(
                    f2_outputs,
                    self.quant_data_dir,
                    f"{self.name}_y",
                )

        return f2_outputs
