from abc import ABC, abstractmethod

import torch

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design_creator import DesignCreator


class DesignCreatorModule(torch.nn.Module, DesignCreator, ABC):
    @abstractmethod
    def create_design(self, name: str) -> Design:
        ...

    @abstractmethod
    def forward(
        self, inputs: torch.FloatTensor, given_inputs_QParams: torch.nn.Module = None
    ) -> torch.FloatTensor:
        ...

    @abstractmethod
    def int_forward(self, q_inputs: torch.IntTensor) -> torch.IntTensor:
        ...
