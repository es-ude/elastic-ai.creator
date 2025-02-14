import logging
from pathlib import Path

import torch
from torch import nn

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.ffn.design import FFN as FFNDesign
from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.relu import ReLU


class FeedForwardNetwork(DesignCreatorModule, nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.num_dimensions = kwargs.get("num_dimensions")
        d_model = kwargs.get("d_model")
        window_size = kwargs.get("window_size")

        device = kwargs.get("device")
        self.name = kwargs.get("name")
        self.quant_bits = kwargs.get("quant_bits")
        self.quant_data_file_dir = kwargs.get("quant_data_file_dir")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.fc1 = (
            Linear(  # TODO: missing num_dimensions(=window_size) for VHDL templates
                name=self.name + "_fc1",
                in_features=d_model,
                out_features=d_model * 4,
                quant_bits=self.quant_bits,
                device=device,
                bias=True,
            )
        )

        self.relu = ReLU(
            name=self.name + "_relu", quant_bits=self.quant_bits, device=device
        )

        self.fc2 = (
            Linear(  # TODO: missing num_dimensions(=window_size) for VHDL templates
                name=self.name + "_fc2",
                in_features=d_model * 4,
                out_features=d_model,
                quant_bits=self.quant_bits,
                device=device,
                bias=True,
            )
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
            fc1=self.fc1,
            relu=self.relu,
            fc2=self.fc2,
            work_library_name="work",
            resource_option="auto",
        )

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def precompute(self):
        self.fc1.precompute()
        self.fc2.precompute()
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.IntTensor,
    ) -> torch.IntTensor:
        assert self.precomputed, "Precompute the model before running int_forward"
        assert not self.training, "int_forward() can only be used in inference mode"

        self._save_quant_data(
            q_inputs, self.quant_data_file_dir, f"{self.fc1.name}_q_x"
        )
        q_f1_outputs = self.fc1.int_forward(
            q_inputs=q_inputs,
        )
        self._save_quant_data(
            q_f1_outputs, self.quant_data_file_dir, f"{self.fc1.name}_q_y"
        )

        self._save_quant_data(
            q_f1_outputs, self.quant_data_file_dir, f"{self.relu.name}_q_x"
        )
        q_relu_outputs = self.relu.int_forward(q_inputs=q_f1_outputs)
        self._save_quant_data(
            q_relu_outputs, self.quant_data_file_dir, f"{self.relu.name}_q_y"
        )

        self._save_quant_data(
            q_relu_outputs, self.quant_data_file_dir, f"{self.fc2.name}_q_x"
        )
        q_f2_outputs = self.fc2.int_forward(
            q_inputs=q_relu_outputs,
        )
        self._save_quant_data(
            q_f2_outputs, self.quant_data_file_dir, f"{self.fc2.name}_q_y"
        )

        return q_f2_outputs

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: nn.Module = None
    ) -> torch.FloatTensor:
        if self.training:
            if given_inputs_QParams is not None:
                self.inputs_QParams = given_inputs_QParams
            else:
                self.inputs_QParams.update_quant_params(inputs)

        f1_outputs = self.fc1(inputs=inputs, given_inputs_QParams=self.inputs_QParams)

        relu_outputs = self.relu(
            inputs=f1_outputs, given_inputs_QParams=self.fc1.outputs_QParams
        )
        f2_outputs = self.fc2(
            inputs=relu_outputs,
            given_inputs_QParams=self.fc1.outputs_QParams,
        )
        self.outputs_QParams = self.fc2.outputs_QParams
        return f2_outputs
