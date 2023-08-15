from abc import ABC, abstractmethod

from elasticai.creator.vhdl.design.design import Design


class Translatable(ABC):
    @abstractmethod
    def translate(self, name: str) -> Design:
        ...
