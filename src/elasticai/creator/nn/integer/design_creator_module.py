from abc import ABC, abstractmethod

import torch

from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.vhdl.design.design import Design


class DesignCreatorModule(ABC, torch.nn.Module):
    @abstractmethod
    def create_design(self, name: str) -> Design: ...

    @abstractmethod
    def forward(
        self, inputs: torch.Tensor, given_inputs_QParams: QParams | None = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor: ...
