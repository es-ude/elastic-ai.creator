from elasticai.creator.hdl_ir import collect_transitive_implementation_closure
from elasticai.creator.ir import attribute, compose_rules

from .ir2vhdl import (
    Code,
    DataGraph,
    Edge,
    Ir2Vhdl,
    IrFactory,
    PluginLoader,
    PluginSpec,
    PluginSymbol,
    Registry,
    ShapeTuple,
    type_handler,
    type_handler_iterable,
)
from .language import (
    Instance,
    InstanceFactory,
    LogicSignal,
    LogicVectorSignal,
    Node,
    NullDefinedLogicSignal,
    PortMap,
    Shape,
    Signal,
)
from .vhdl_template import (
    EntityTemplateDirector,
    EntityTemplateParameter,
    ValueTemplateParameter,
)

factory = IrFactory()

__all__ = [
    "Ir2Vhdl",
    "Edge",
    "Code",
    "Node",
    "attribute",
    "DataGraph",
    "Signal",
    "Instance",
    "factory",
    "InstanceFactory",
    "LogicSignal",
    "LogicVectorSignal",
    "NullDefinedLogicSignal",
    "IrFactory",
    "PluginLoader",
    "PluginSpec",
    "PluginSymbol",
    "PortMap",
    "Registry",
    "Shape",
    "ShapeTuple",
    "type_handler",
    "type_handler_iterable",
    "EntityTemplateParameter",
    "ValueTemplateParameter",
    "EntityTemplateDirector",
    "collect_transitive_implementation_closure",
    "compose_rules",
]
