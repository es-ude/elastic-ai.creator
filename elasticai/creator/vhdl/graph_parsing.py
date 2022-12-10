from typing import Set

from mlframework import Module
from vhdl.model_tracing import Graph, Node


def collect_modules(g: Graph) -> Set[Module]:
    modules = set()
    for node in filter(_is_module_node, g.nodes):
        modules.add(node.module)
    return modules


def _is_module_node(node: Node) -> bool:
    return hasattr(node, "module")
