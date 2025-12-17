from typing import Protocol


class NodeConstraintFn[PN, P](Protocol):
    def __call__(self, pattern_node: PN, graph_node: P) -> bool: ...
