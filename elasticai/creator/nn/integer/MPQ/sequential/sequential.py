from pathlib import Path
from typing import Any, cast

import torch

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.sequential.design_builder import SequentialDesignBuilder
from elasticai.creator.vhdl.design.design import Design

from .design import Sequential as _SequentialDesign


class _MPQSequentialDesignFactory:
    def create_sequential_design(self, sub_designs: list[Design], name: str) -> Design:
        return _SequentialDesign(sub_designs=sub_designs, name=name)


class Sequential(DesignCreatorModule, torch.nn.Sequential):
    def __init__(
        self, *submodules: DesignCreatorModule, name: str, quant_data_dir: Path
    ) -> None:
        super().__init__(*submodules)
        self.submodules: tuple[Any, ...] = submodules
        self.name = name
        self.quant_data_dir = quant_data_dir
        self.precomputed = False
        self._design_builder = SequentialDesignBuilder(_MPQSequentialDesignFactory())
        # Set quant_data_dir on each submodule
        for submodule in submodules:
            cast(Any, submodule).quant_data_dir = quant_data_dir

    def forward(
        self,
        inputs: torch.FloatTensor,
        given_inputs_QParams: torch.nn.Module | None = None,
        enable_simquant: bool = True,
    ) -> torch.FloatTensor:
        outputs = inputs
        for submodule in self.submodules:
            outputs = submodule(
                outputs,
                given_inputs_QParams=given_inputs_QParams,
                enable_simquant=enable_simquant,
            )
            if enable_simquant:
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
        from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
            save_quant_data,
        )

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")
        q_outputs = q_inputs
        for submodule in self.submodules:
            q_outputs = submodule.int_forward(q_outputs)
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def create_design(self, name: str) -> Design:
        return self._design_builder.build(self.submodules, name)
