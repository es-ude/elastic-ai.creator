"""HDL (Hardware Description Language) Generator Abstraction.

This module provides a unified interface for generating VHDL and Verilog code.
It abstracts away the differences between the two languages, allowing you to write
code generation logic once and target either HDL.

The abstraction uses a Protocol to define the interface, with private concrete
implementations for each language. Use create_generator() to obtain a generator.

Example:
    ```python
    from elasticai.creator.hdl_generator import create_generator, HDLLanguage

    # Create a generator for VHDL - returns HDLGenerator protocol type
    generator = create_generator(HDLLanguage.VHDL)

    # Create signals/wires - returns HDLSignal protocol type
    clk = generator.create_signal("clk", width=1)
    data = generator.create_signal("data", width=8)

    # Instantiate a module/entity - returns HDLInstance protocol type
    instance = generator.create_instance(
        node=my_node,
        generics={"WIDTH": "8"},
        ports={"clk": clk, "data": data}
    )

    # Generate code
    for line in instance.instantiate():
        print(line)
    ```
"""

from .factory import HDLGenerator, HDLLanguage, create_generator
from .protocols import (
    HDLInstance,
    HDLSignal,
    HDLTemplateDirector,
)

__all__ = [
    "create_generator",
    "HDLGenerator",
    "HDLLanguage",
    "HDLInstance",
    "HDLSignal",
    "HDLTemplateDirector",
]
