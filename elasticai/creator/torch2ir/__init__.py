"""Transform PyTorch models to IR.

The translation process is highly customizable.
(`Torch2Ir`)[elasticai.creator.torch2ir.Torch2Ir] is responsible
for the translation process and will call *module handlers* to
extract attributes from modules. These attributes are then
attached to the corresponding nodes in the IR.

Each module handler is a function that takes a PyTorch module
and returns a dictionary with the extracted attributes.

The `Torch2Ir` class features some factory methods as class
methods, e.g., (`Torch2Ir.get_default_converter`)[elasticai.creator.torch2ir.Torch2Ir.get_default_converter].
Those will create a new `Torch2Ir` instance and register some
preconfigured module handlers.
However, you are free to extend or alter the behaviour of the
translation process by registering your own module handlers.
"""

from elasticai.creator.ir import edge as new_edge

from .core import Edge, Implementation, Node, input_node, new_node, output_node
from .default_handlers import handlers as default_module_handlers
from .torch2ir import Torch2Ir, get_default_converter

__all__ = [
    "Torch2Ir",
    "get_default_converter",
    "Implementation",
    "Node",
    "Edge",
    "new_node",
    "new_edge",
    "input_node",
    "output_node",
    "get_default_converter",
    "default_module_handlers",
]
