"""Protocol definitions for HDL abstraction.

This module defines the protocols (interfaces) that all HDL implementations must follow.
Using Protocol classes allows for structural subtyping (duck typing with type checking).
"""

from abc import abstractmethod
from collections.abc import Iterator
from string import Template
from typing import Protocol, runtime_checkable


@runtime_checkable
class HDLSignal(Protocol):
    """Protocol for HDL signals/wires.
    
    Represents either a VHDL signal or a Verilog wire.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the signal/wire."""
        ...

    @abstractmethod
    def define(self) -> Iterator[str]:
        """Generate HDL code lines to define this signal/wire."""
        ...

    @abstractmethod
    def make_instance_specific(self, instance: str) -> "HDLSignal":
        """Create a signal/wire with an instance-specific name.
        
        Args:
            instance: The instance name to append to the signal name.
            
        Returns:
            A new signal with the modified name.
        """
        ...


@runtime_checkable
class HDLInstance(Protocol):
    """Protocol for HDL module/entity instances.
    
    Represents either a VHDL entity instance or a Verilog module instance.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The instance name."""
        ...

    @property
    @abstractmethod
    def implementation(self) -> str:
        """The name of the module/entity being instantiated."""
        ...

    @abstractmethod
    def define_signals(self) -> Iterator[str]:
        """Generate HDL code lines to define all signals/wires used by this instance."""
        ...

    @abstractmethod
    def instantiate(self) -> Iterator[str]:
        """Generate HDL code lines to instantiate this module/entity."""
        ...


@runtime_checkable
class HDLTemplateDirector(Protocol):
    """Protocol for HDL template directors.
    
    Provides a builder interface for creating HDL templates from prototype code.
    """

    @abstractmethod
    def set_prototype(self, prototype: str) -> "HDLTemplateDirector":
        """Set the prototype HDL code to use as a template base.
        
        Args:
            prototype: The HDL code to convert into a template.
            
        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def add_parameter(self, name: str) -> "HDLTemplateDirector":
        """Add a template parameter (generic/parameter).
        
        Args:
            name: The name of the parameter to make templatable.
            
        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def build(self) -> Template:
        """Build the final template.
        
        Returns:
            A string.Template that can be used to generate code.
        """
        ...
