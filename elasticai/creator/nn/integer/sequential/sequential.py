from pathlib import Path

import torch

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.sequential.layer import Sequential as _SequentialBase
from elasticai.creator.vhdl.design.design import Design

from .design import Sequential as _SequentialDesign


class Sequential(_SequentialBase):
    def __init__(
        self, *submodules: DesignCreatorModule, name: str, quant_data_file_dir: Path
    ) -> None:
        super().__init__(*submodules)
        self.submodules = submodules
        self.name = name
        self.quant_data_file_dir = quant_data_file_dir
        self.precomputed = False

    def _save_quant_data(self, tensor, file_dir: Path, file_name: str):
        file_path = file_dir / f"{file_name}.txt"
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        file_path.write_text(tensor_str)

    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        outputs = inputs
        for submodule in self.submodules:
            # outputs = submodule(outputs)
            outputs = submodule(outputs, given_inputs_QParams=given_inputs_QParams)
            given_inputs_QParams = submodule.outputs_QParams
        return outputs

    def quantize_inputs(self, inputs: torch.FloatTensor) -> torch.IntTensor:
        return self.submodules[0].inputs_QParams.quantize(inputs)

    def dequantize_outputs(self, q_outputs: torch.IntTensor) -> torch.FloatTensor:
        return self.submodules[-1].outputs_QParams.dequantize(q_outputs)

    def precompute(self):
        for submodule in self.submodules:
            if hasattr(submodule, "precompute"):
                submodule.precompute()
        self.precomputed = True

    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        assert not self.training, "int_forward() should only be called in eval mode"
        assert self.precomputed, "precompute() should be called before int_forward()"
        self._save_quant_data(q_inputs, self.quant_data_file_dir, f"{self.name}_q_x")
        # print(f"-------------------{self.name}-------------------")
        # i = 0
        for submodule in self.submodules:
            self._save_quant_data(
                q_inputs, self.quant_data_file_dir, f"{submodule.name}_q_x"
            )
            q_outputs = submodule.int_forward(q_inputs)

            q_inputs = q_outputs
            self._save_quant_data(
                q_outputs, self.quant_data_file_dir, f"{submodule.name}_q_y"
            )
            # i += 1
            # if i == 4:
            #     break
        self._save_quant_data(q_outputs, self.quant_data_file_dir, f"{self.name}_q_y")
        return q_outputs

    def create_sequential_design(self, sub_designs: list[Design], name: str) -> Design:
        return _SequentialDesign(sub_designs=sub_designs, name=name)
