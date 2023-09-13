from abc import ABC, abstractmethod

from elasticai.creator.vhdl.design.design import Design


class DesignCreator(ABC):
    @abstractmethod
    def create_design(self, name: str) -> Design: ...
