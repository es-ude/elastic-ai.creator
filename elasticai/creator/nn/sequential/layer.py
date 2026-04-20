from typing import cast

import torch

from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.sequential.design_builder import (
    SequentialDesignBuilder,
    SequentialDesignFactory,
)
from elasticai.creator.vhdl.design.design import Design

from .design import Sequential as _SequentialDesign


class _DefaultSequentialDesignFactory:
    def create_sequential_design(self, sub_designs: list[Design], name: str) -> Design:
        return _SequentialDesign(sub_designs=sub_designs, name=name)


class Sequential(DesignCreatorModule, torch.nn.Sequential):
    def __init__(
        self,
        *submodules: DesignCreatorModule,
        design_factory: SequentialDesignFactory | None = None,
    ):
        super().__init__(*submodules)
        self._design_factory = design_factory or _DefaultSequentialDesignFactory()

    def create_design(self, name: str) -> Design:
        submodules = [cast(DesignCreatorModule, module) for module in self.children()]
        return SequentialDesignBuilder(self._design_factory).build(submodules, name)
