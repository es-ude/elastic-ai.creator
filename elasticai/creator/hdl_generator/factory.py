"""HDL generators for VHDL and Verilog code generation.

This module provides concrete implementations for generating VHDL and Verilog code.
Each generator encapsulates language-specific factories and provides a unified interface.
"""

from enum import Enum, auto
from typing import Generic, Literal, Protocol, TypeVar, overload

from elasticai.creator.hdl_ir import Node
from elasticai.creator.ir2verilog import BaseWire
from elasticai.creator.ir2vhdl import Signal

from . import verilog_impl, vhdl_impl
from .protocols import HDLInstance, HDLSignal, HDLTemplateDirector

# TypeVar for signal types - covariant so HDLGenerator[Signal] is a subtype of HDLGenerator[HDLSignal]
TSignal = TypeVar("TSignal", bound=HDLSignal, covariant=True)


# Type alias for generic/parameter dictionaries
type GenericMap = dict[str, str]
type ParameterMap = dict[str, str]
type PortMap = dict[str, HDLSignal]


class HDLLanguage(Enum):
    """Supported HDL languages."""

    VHDL = auto()
    VERILOG = auto()


class HDLGenerator(Protocol, Generic[TSignal]):
    """Protocol defining the interface for HDL code generation.

    This protocol is generic over the signal type to provide static type safety.
    When you create a VHDL generator, signals are typed as VHDL Signal objects.
    When you create a Verilog generator, signals are typed as Verilog BaseWire objects.

    This prevents accidentally mixing VHDL signals with Verilog instances (and vice versa)
    at compile time, catching errors before runtime.

    Example:
        ```python
        from elasticai.creator.hdl_generator import create_generator, HDLLanguage
        from elasticai.creator.hdl_ir import Node, Shape

        # Create a generator for VHDL - returns HDLGenerator[Signal]
        vhdl_gen = create_generator(HDLLanguage.VHDL)

        # Create signals - returns Signal (VHDL-specific type)
        clk = vhdl_gen.create_signal("clk", width=1)
        data_in = vhdl_gen.create_signal("data_in", width=8)

        # Create a Verilog generator - returns HDLGenerator[BaseWire]
        verilog_gen = create_generator(HDLLanguage.VERILOG)

        # This would be a TYPE ERROR caught by the type checker:
        # vhdl_gen.create_instance(ports={"clk": verilog_gen.create_signal("clk")})
        ```
    """

    def create_signal(self, name: str, width: int | None = None) -> TSignal:
        """Create a signal/wire.

        Creates a VHDL signal or Verilog wire depending on the language.

        Args:
            name: The signal/wire name.
            width: The bit width. If None or 1, creates a single-bit signal.
                   Otherwise creates a vector signal/wire.

        Returns:
            A language-specific signal that can be used with create_instance.

        Example:
            ```python
            gen = VHDLGenerator()
            clk = gen.create_signal("clk")  # Single bit
            data = gen.create_signal("data", width=8)  # 8-bit vector
            ```
        """
        ...

    def create_null_signal(self, name: str) -> TSignal:
        """Create a signal/wire that doesn't need definition.

        Useful for input ports or signals that are defined elsewhere.

        Args:
            name: The signal/wire name.

        Returns:
            A language-specific signal that produces no definition code.
        """
        ...

    def create_instance(
        self,
        node: Node,
        generics: dict[str, str] | None = None,
        ports: dict[str, TSignal] | None = None,
    ) -> HDLInstance:
        """Create a module/entity instance.

        Args:
            node: The node representing the module/entity to instantiate.
            generics: Generic/parameter values (name -> value). Optional.
            ports: Port connections (port_name -> signal). Must be signals
                   created by the same generator. Optional.

        Returns:
            An HDL instance that can generate instantiation code.

        Note:
            The type system ensures you can only pass signals created by the
            same generator. Mixing VHDL signals with a Verilog generator
            (or vice versa) will be caught as a type error.
        """
        ...

    def create_template_director(self) -> HDLTemplateDirector:
        """Create a template director for building HDL templates.

        Template directors allow you to convert prototype HDL code into
        parameterized templates.

        Returns:
            An HDL template director for building templates.

        Example:
            ```python
            gen = VHDLGenerator()

            prototype_code = '''
            entity adder is
                generic (WIDTH : natural := 8);
                port (
                    a, b : in std_logic_vector(WIDTH-1 downto 0);
                    sum : out std_logic_vector(WIDTH-1 downto 0)
                );
            end entity;
            '''

            template = (
                gen.create_template_director()
                .set_prototype(prototype_code)
                .add_parameter("WIDTH")
                .build()
            )

            # Use the template
            code = template.substitute(entity="my_adder", WIDTH="16")
            ```
        """
        ...


class _VHDLGenerator:
    """VHDL implementation of HDL generator.

    Private class - use create_generator() instead to ensure proper abstraction.
    """

    def create_signal(self, name: str, width: int | None = None) -> Signal:
        """Create a VHDL signal.

        Args:
            name: The signal name.
            width: The bit width. If None or 1, creates a single-bit signal.
                   Otherwise creates a vector signal.

        Returns:
            A VHDL signal.
        """
        return vhdl_impl.create_signal(name, width)

    def create_null_signal(self, name: str) -> Signal:
        """Create a VHDL signal that doesn't need definition.

        Args:
            name: The signal name.

        Returns:
            A VHDL null signal.
        """
        return vhdl_impl.create_null_signal(name)

    def create_instance(
        self,
        node: Node,
        generics: dict[str, str] | None = None,
        ports: dict[str, Signal] | None = None,
    ) -> HDLInstance:
        """Create a VHDL entity instance.

        Args:
            node: The node representing the entity to instantiate.
            generics: Generic values (name -> value). Optional.
            ports: Port connections (port_name -> signal). Optional.

        Returns:
            A VHDL instance.
        """
        if generics is None:
            generics = {}
        if ports is None:
            ports = {}
        return vhdl_impl.create_instance(node, generics, ports)

    def create_template_director(self) -> HDLTemplateDirector:
        """Create a VHDL template director.

        Returns:
            A VHDL template director.
        """
        return vhdl_impl.VHDLTemplateDirector()


class _VerilogGenerator:
    """Verilog implementation of HDL generator.

    Private class - use create_generator() instead to ensure proper abstraction.
    """

    def create_signal(self, name: str, width: int | None = None) -> BaseWire:
        """Create a Verilog wire.

        Args:
            name: The wire name.
            width: The bit width. If None or 1, creates a single-bit wire.
                   Otherwise creates a vector wire.

        Returns:
            A Verilog wire.
        """
        return verilog_impl.create_signal(name, width)

    def create_null_signal(self, name: str) -> BaseWire:
        """Create a Verilog wire that doesn't need definition.

        Args:
            name: The wire name.

        Returns:
            A Verilog null wire.
        """
        return verilog_impl.create_null_signal(name)

    def create_instance(
        self,
        node: Node,
        generics: dict[str, str] | None = None,
        ports: dict[str, BaseWire] | None = None,
    ) -> HDLInstance:
        """Create a Verilog module instance.

        Args:
            node: The node representing the module to instantiate.
            generics: Parameter values (name -> value). Optional.
            ports: Port connections (port_name -> wire). Optional.

        Returns:
            A Verilog instance.
        """
        if generics is None:
            generics = {}
        if ports is None:
            ports = {}
        return verilog_impl.create_instance(node, generics, ports)

    def create_template_director(self) -> HDLTemplateDirector:
        """Create a Verilog template director.

        Returns:
            A Verilog template director.
        """
        return verilog_impl.VerilogTemplateDirector()


@overload
def create_generator(language: Literal[HDLLanguage.VHDL]) -> HDLGenerator[Signal]: ...


@overload
def create_generator(
    language: Literal[HDLLanguage.VERILOG],
) -> HDLGenerator[BaseWire]: ...


def create_generator(
    language: HDLLanguage,
) -> HDLGenerator[Signal] | HDLGenerator[BaseWire]:
    """Create an HDL generator for the specified language.

    This factory function provides static type safety through overloads:
    - When you pass HDLLanguage.VHDL, you get HDLGenerator[Signal]
    - When you pass HDLLanguage.VERILOG, you get HDLGenerator[BaseWire]

    This means the type checker can verify that you don't mix VHDL signals
    with Verilog instances (or vice versa) at compile time.

    Args:
        language: The target HDL language (VHDL or VERILOG).

    Returns:
        An HDL generator with the appropriate signal type.

    Example:
        ```python
        from elasticai.creator.hdl_generator import create_generator, HDLLanguage

        # Create VHDL generator - typed as HDLGenerator[Signal]
        vhdl_gen = create_generator(HDLLanguage.VHDL)
        vhdl_signal = vhdl_gen.create_signal("data", width=8)  # Returns Signal

        # Create Verilog generator - typed as HDLGenerator[BaseWire]
        verilog_gen = create_generator(HDLLanguage.VERILOG)
        verilog_wire = verilog_gen.create_signal("data", width=8)  # Returns BaseWire

        # Type checker will catch this error:
        # vhdl_gen.create_instance(ports={"data": verilog_wire})  # TYPE ERROR!
        ```
    """
    if language == HDLLanguage.VHDL:
        return _VHDLGenerator()
    elif language == HDLLanguage.VERILOG:
        return _VerilogGenerator()
    else:
        raise ValueError(f"Unsupported HDL language: {language}")
