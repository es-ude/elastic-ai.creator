from pathlib import Path

import torch

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.quant_utils.SaveQuantData import save_quant_data
from elasticai.creator.nn.sequential.layer import Sequential as _SequentialBase

from .design import Sequential as _SequentialDesign


class Sequential(_SequentialBase):
    def __init__(self, *submodules: DesignCreatorModule):
        super().__init__(*submodules)
        self.submodules = submodules

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        given_input_QParams = None

        for submodule in self.submodules:
            x = submodule(x, given_input_QParams=given_input_QParams)
            given_input_QParams = submodule.output_QParams

        return x

    def int_forward(
        self, inputs: torch.Tensor, quant_data_file_dir: Path, name: str
    ) -> torch.Tensor:
        assert not self.training, "int_forward() should only be called in eval mode"

        x = inputs

        # Save quantized input to file
        if x.dtype != torch.int32:
            x = self.submodules[0].input_QParams.quantizeProcess(x)
            if quant_data_file_dir is not None:
                save_quant_data(x, quant_data_file_dir, f"{name}_q_x")

        for submodule in self.submodules:
            x = submodule.int_forward(x, quant_data_file_dir)

        if x.dtype == torch.int32 and quant_data_file_dir is not None:
            save_quant_data(x, quant_data_file_dir, f"{name}_q_y")

        x = self.submodules[-1].output_QParams.dequantizeProcess(x)

        return x

    def create_sequential_design(self, sub_designs: list[Design], name: str) -> Design:
        return _SequentialDesign(sub_designs=sub_designs, name=name)
