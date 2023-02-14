from typing import Iterable, Optional, Reversible

from torch.fx import Graph as fxGraph

from elasticai.creator.vhdl.tracing.typing import (
    HWEquivalentGraph,
    Node,
    TranslatableLayer,
)


class _HWEquivalentGraph(HWEquivalentGraph[TranslatableLayer]):
    """
        The HWEquivalentGraph is the result of tracing a compatible neural network `m`
    with the corresponding HWEquivalentTracer. It combines signal and
    port maps for instantiation for all nodes linked to HWEquivalent submodules of `m`
    by making calls to these submodules.
    """

    def __init__(
        self, fx_graph: fxGraph, modules_by_nodes: dict[str, TranslatableLayer]
    ):
        self._fx_graph = fx_graph
        self._modules_by_nodes = modules_by_nodes

    @property
    def nodes(self) -> Reversible[Node]:
        return self._fx_graph.nodes

    def node_has_module(self, node: str | Node) -> bool:
        if isinstance(node, str):
            return node in self._modules_by_nodes
        return node.name in self._modules_by_nodes

    def get_module_for_node(self, node: str | Node) -> Optional[TranslatableLayer]:
        if isinstance(node, str):
            return self._modules_by_nodes[node]
        return self._modules_by_nodes[node.name]

    @property
    def module_nodes(self) -> Iterable[Node]:
        return filter(lambda n: n.name in self._modules_by_nodes, self._fx_graph.nodes)
