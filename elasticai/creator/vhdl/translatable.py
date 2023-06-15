from abc import ABC, abstractmethod

from elasticai.creator.vhdl.design_base.design import Design


class Translatable(ABC):
    @abstractmethod
    def translate(self, name: str) -> Design:
        ...
