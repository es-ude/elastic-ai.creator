import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.MPQ.ffn.design import FFN as FFNDesign
from elasticai.creator.nn.integer.MPQ.linear import Linear
from elasticai.creator.nn.integer.MPQ.relu import ReLU
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class FeedForwardNetwork(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        d_model = kwargs.get("d_model")
        ffn_dim = kwargs.get("ffn_dim")
        num_dimensions = kwargs.get("num_dimensions")

        self.name = kwargs.get("name")
        self.quantizable_elements = ["inputs", "outputs"]
        self.quant_bits_per_element = None
        self.quant_data_dir = kwargs.get("quant_data_dir", None)
        self.device = kwargs.get("device")
        self.enable_error_analysis = kwargs.get("enable_error_analysis", False)

        self.MPQ_strategy = kwargs.get("MPQ_strategy")

        self.fc1 = Linear(
            name=self.name + "_fc1",
            in_features=d_model,
            out_features=ffn_dim,
            num_dimensions=num_dimensions,
            bias=True,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.relu = ReLU(
            name=self.name + "_relu",
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
        )

        self.fc2 = Linear(
            name=self.name + "_fc2",
            in_features=ffn_dim,
            out_features=d_model,
            num_dimensions=num_dimensions,
            bias=True,
            quant_data_dir=self.quant_data_dir,
            device=self.device,
            enable_error_analysis=self.enable_error_analysis,
            MPQ_strategy=self.MPQ_strategy,
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

    def create_design(self, name: str) -> FFNDesign:
        return FFNDesign(
            name=name,
            data_width=self.quant_bits_per_element["inputs"],  # TODO
            fc1=self.fc1,
            relu=self.relu,
            fc2=self.fc2,
            work_library_name="work",
        )

    def precompute(self):
        assert not self.training, "int_forward should be called in eval mode"
        self.fc1.precompute()
        self.relu.precompute()
        self.fc2.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"
        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        q_f1_outputs = self.fc1.int_forward(
            q_inputs=q_inputs,
        )
        q_relu_outputs = self.relu.int_forward(q_inputs=q_f1_outputs)
        q_f2_outputs = self.fc2.int_forward(
            q_inputs=q_relu_outputs,
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

        f1_outputs = self.fc1.forward(
            inputs=inputs,
            given_inputs_QParams=self.inputs_QParams,
            enable_simquant=enable_simquant,
        )

        relu_outputs = self.relu.forward(
            inputs=f1_outputs,
            given_inputs_QParams=self.fc1.outputs_QParams,
            enable_simquant=enable_simquant,
        )
        f2_outputs = self.fc2.forward(
            inputs=relu_outputs,
            given_inputs_QParams=self.fc1.outputs_QParams,
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
