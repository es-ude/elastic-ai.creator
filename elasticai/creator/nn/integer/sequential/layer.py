from pathlib import Path

import torch

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.sequential.layer import Sequential as _SequentialBase
from elasticai.creator.vhdl.design.design import Design

from .design import Sequential as _SequentialDesign


class Sequential(_SequentialBase):
    def __init__(self, *submodules: DesignCreatorModule):
        super().__init__(*submodules)
        self.submodules = submodules

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        given_input_QParams = None

        for submodule in self.submodules:
            x = submodule(x, given_input_QParams=given_input_QParams)
            given_input_QParams = submodule.output_QParams

        return x

    def int_forward(
        self, input: torch.FloatTensor, quant_data_file_dir: Path, name: str
    ) -> torch.FloatTensor:
        assert not self.training, "int_forward() should only be called in eval mode"

        q_input = self.submodules[0].input_QParams.quantize(input)
        if quant_data_file_dir is not None:
            self._save_quant_data(q_input, quant_data_file_dir, f"{name}_q_x")

        for submodule in self.submodules:
            self._save_quant_data(q_input, quant_data_file_dir, f"{submodule.name}_q_x")
            q_output = submodule.int_forward(q_input)
            self._save_quant_data(
                q_output, quant_data_file_dir, f"{submodule.name}_q_y"
            )
            q_input = q_output

        self._save_quant_data(q_output, quant_data_file_dir, f"{name}_q_y")

        dq_output = self.submodules[-1].output_QParams.dequantize(q_output)

        return dq_output

    def create_sequential_design(self, sub_designs: list[Design], name: str) -> Design:
        return _SequentialDesign(sub_designs=sub_designs, name=name)
