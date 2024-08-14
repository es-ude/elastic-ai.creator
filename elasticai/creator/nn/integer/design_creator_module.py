from abc import ABC, abstractmethod
from pathlib import Path

import torch

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design_creator import DesignCreator


class DesignCreatorModule(torch.nn.Module, DesignCreator, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create_design(self, name: str) -> Design: ...

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def int_forward(
        self, inputs: torch.Tensor, quant_data_file_dir: Path
    ) -> torch.Tensor: ...
