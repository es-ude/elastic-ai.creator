from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from elasticai.creator.vhdl.hardware_description_language.signals import Signal

T_Design = TypeVar("T_Design", bound="Design")


@dataclass
class Port:
    in_signals: set[Signal]
    out_signals: set[Signal]

    @property
    def signals(self) -> set[Signal]:
        return self.in_signals | self.out_signals


class Design(Generic[T_Design], ABC):
    """
    Represents a hardware_description_language design. As such it typically corresponds to one or more vhdl/verilog files.
    The `Port` object exposed through `Design.port` represents the port definition of the design, to enable
    programmatic connection of different hardware_description_language designs. The class will be typically combined with something
    that represents a file or a collection of files to mirror the generated code.

    The purpose of this class is to facilitate auto-wiring of different designs and holding names to generate file/folder
    names.

    For Programmers: this is similar to a class in an OO programming language
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            self._name = f"{self.__class__.__name__}"
        else:
            self._name = name

    @property
    def name(self) -> str:
        """
        This should be the same as the name of the file or folder containing the HDL code that belongs to this design.
        """
        return self._name

    @property
    @abstractmethod
    def port(self) -> Port:
        """
        The port as defined by the top level module/entity in this design.
        """
        ...

    def instantiate(self: T_Design, name: str) -> "Instance[T_Design]":
        """
        Returns an instance of the design with the name `name`. The instance will contain a reference to the original
        design allowing.
        """
        return Instance(design=self, name=name)


@dataclass
class Instance(Generic[T_Design]):
    design: T_Design
    name: str
