import copy
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Generic, Self, TypeVar, overload

from elasticai.creator.graph import Graph, NodeConstraintFn, find_all_subgraphs, rewrite
from elasticai.creator.ir.base.attribute import Attribute

from ..core import Edge, Implementation, Node

N = TypeVar("N", bound=Node)
E = TypeVar("E", bound=Edge)


class RemappedSubImplementation(Generic[N]):
    """Use a given mapping to make the data dictionary accessible via node names from graph.

    This allows to access the data dictionary using the names from the graph.
    The provided mapping is used to map the node names of the graph to the node names in the data dictionary.
    This is used e.g., to initialize new nodes, that were replaced during rewriting.
    The mapping is assumed to be one-to-one, i.e. each node name in the graph maps to exactly one node name in the data dictionary.

    """

    def __init__(
        self,
        mapping: dict[str, str],
        graph: Graph[str],
        data: dict[str, Attribute],
        node_fn: Callable[[str, dict[str, Attribute]], N],
    ):
        self._mapping = mapping
        self._inverted = dict(((v, k) for k, v in mapping.items()))
        self._data = data
        self._graph = graph
        self._node_constr = node_fn

    def _original_name(self, node: str):
        return self._mapping[node]

    def _mapped_name(self, node: str):
        return self._inverted[node]

    @property
    def nodes(self) -> Mapping[str, N]:
        return self._create_remapped_data(self._graph.nodes)

    def successors(self, node: str) -> Mapping[str, N]:
        return self._create_remapped_data(
            self._graph.successors[self._inverted[node]],
        )

    def _create_remapped_data(self, iterable: Iterable[str]) -> Mapping[str, N]:
        return _RemappedData(
            self._mapping, self._data, iterable, node_fn=self._node_constr
        )

    def predecessors(self, node: str) -> Mapping[str, N]:
        return self._create_remapped_data(
            self._graph.predecessors[self._inverted[node]]
        )


@overload
def remap_sub_implementation(
    mapping: dict[str, str],
    graph: Graph[str],
    data: dict[str, Attribute],
    node_fn: Callable[[str, dict[str, Attribute]], N],
) -> RemappedSubImplementation[N]: ...


@overload
def remap_sub_implementation(
    mapping: dict[str, str],
    graph: Graph[str],
    data: dict[str, Attribute],
) -> RemappedSubImplementation[Node]: ...


def remap_sub_implementation(
    mapping: dict[str, str],
    graph: Graph[str],
    data: dict[str, Attribute],
    node_fn: Callable[[str, dict[str, Attribute]], N] | None = None,
) -> RemappedSubImplementation[N] | RemappedSubImplementation[Node]:
    return RemappedSubImplementation(
        mapping=mapping, graph=graph, data=data, node_fn=node_fn or Node
    )


class _RemappedData(Mapping[str, N]):
    def __init__(
        self,
        remapping: dict[str, str],
        data: dict[str, Any],
        iterable: Iterable[str],
        node_fn: Callable[[str, dict[str, Any]], N],
    ) -> None:
        self._remapping = remapping
        self._inverted_remapping = dict(((v, k) for k, v in self._remapping.items()))
        self._data = data
        self._iter = iterable
        self._node_constr = node_fn

    def __getitem__(self, key) -> N:
        return self._node_constr(key, self._data[self._remapping[key]])

    def __len__(self) -> int:
        return len(self._remapping)

    def __iter__(self) -> Iterator[Any]:
        for key in iter(self._iter):
            if key in self:
                yield key

    def __contains__(self, key: Any) -> bool:
        return key in self._remapping


class RewritingContext:
    def __init__(
        self, match, rule: "RewriteRule", original_impl, replacement_map, new_impl
    ):
        self.match = RemappedSubImplementation(
            mapping=match,
            graph=rule.pattern.graph,
            data=original_impl.data["nodes"],  # type: ignore
            node_fn=Node,
        )
        self.replacement = RemappedSubImplementation(
            mapping=replacement_map,
            graph=rule.replacement.graph,
            data=new_impl.data["nodes"],  # type: ignore
            node_fn=Node,
        )


class Rewriter:
    def __init__(self):
        self._rules = []

    def add_rule(self, rule: "RewriteRule") -> Self:
        self._rules.append(rule)
        return self

    def _lift_constraint_fn(
        self,
        fn: Callable[[Node, Node], bool],
        pattern: Implementation,
        impl: Implementation,
    ) -> NodeConstraintFn[str, str]:
        def lifted(pattern_node: str, graph_node: str) -> bool:
            return fn(pattern.nodes[pattern_node], impl.nodes[graph_node])

        return lifted

    def _apply_rule(self, impl: Implementation, rule: "RewriteRule") -> Implementation:
        matches = find_all_subgraphs(
            graph=impl.graph,
            pattern=rule.pattern.graph,
            node_constraint=self._lift_constraint_fn(
                fn=rule.node_constraint, impl=impl, pattern=rule.pattern
            ),
        )
        replacement_maps = []
        rewritten = impl.graph
        for match in matches:
            rewritten, replacement_map = rewrite(
                replacement=rule.replacement.graph,
                original=rewritten,
                match=match,
                lhs={x: x for x in rule.interface},
                rhs={x: x for x in rule.interface},
            )
            replacement_maps.append(replacement_map)
        new_impl = Implementation(graph=rewritten, data=copy.deepcopy(impl.data))
        new_impl.sync_data_with_graph()
        contexts = []
        for match, replacement_map in zip(matches, replacement_maps):
            context = RewritingContext(
                replacement_map=replacement_map,
                match=match,
                rule=rule,
                original_impl=impl,
                new_impl=new_impl,
            )
            contexts.append(context)

        for ctx in contexts:
            for node_name in ctx.replacement.nodes:
                if node_name not in rule.interface:
                    initialize = rule.replacement.nodes[node_name].attributes["init"]
                    node = ctx.replacement.nodes[node_name]
                    node.data.update(initialize(ctx.match))

        return new_impl

    def apply(self, impl: Implementation) -> Implementation:
        for rule in self._rules:
            impl = self._apply_rule(impl, rule)
        return impl


class RewriteRule:
    def __init__(
        self,
        pattern,
        replacement,
        node_constraint,
        interface,
    ):
        self.pattern = pattern
        self.replacement = replacement
        self.node_constraint = node_constraint
        self.interface = interface
