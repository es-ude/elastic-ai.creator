from abc import ABC, abstractmethod

import torch

from elasticai.creator.vhdl.design.design import Design


class DesignCreatorModule(ABC, torch.nn.Module):
    @abstractmethod
    def create_design(self, name: str) -> Design: ...
