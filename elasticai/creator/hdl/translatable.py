from abc import ABC, abstractmethod

from elasticai.creator.hdl.design_base.design import Design


class Translatable(ABC):
    @abstractmethod
    def translate(self, name: str) -> Design:
        ...
