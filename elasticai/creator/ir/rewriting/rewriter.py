import copy
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Generic, Self, TypeVar, overload

from elasticai.creator.graph import (
    Graph,
    NodeConstraintFn,
    find_all_subgraphs,
    get_rewriteable_matches,
    rewrite,
)
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


class _RewritingContext:
    def __init__(
        self,
        match,
        rule: "RewriteRule",
        original_impl,
        replacement_map=None,
        new_impl=None,
    ):
        self.replacement_map = replacement_map
        self.matched_impl = RemappedSubImplementation(
            mapping=match,
            graph=rule.pattern.graph,
            data=original_impl.data["nodes"],  # type: ignore
            node_fn=Node,
        )
        self.match = match
        self.replacement = rule.replacement(self.matched_impl)
        self._new_impl: Implementation | None = new_impl

    @property
    def new_impl(self) -> Implementation:
        if self._new_impl is None:
            raise ValueError("New implementation has not been set yet.")
        return self._new_impl

    @new_impl.setter
    def new_impl(self, value: Implementation) -> None:
        if self.replacement_map is None:
            raise ValueError("Replacement map has not been set yet.")
        self._new_impl = RemappedSubImplementation(
            mapping=self.replacement_map,
            graph=self.replacement.graph,
            data=value.data["nodes"],  # type: ignore
            node_fn=Node,
        )


class Rewriter:
    """Apply list of `RewriteRule`s to an `Implementation`.

    The result is a new implementation. The original implementation is not modified.
    The rules are applied in the order they were added.
    For more information on how to create rules, see `RewriteRule`.
    """

    def __init__(self) -> None:
        self._rules: list["RewriteRule"] = []
        self._current_rule: "RewriteRule" | None = None
        self._current_impl: Implementation | None = None
        self._current_contexts: list[_RewritingContext] = []

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
        self._current_rule = rule
        self._current_impl = impl
        self._current_contexts = []
        matches: Iterable[dict[str, str]] = find_all_subgraphs(
            graph=impl.graph,
            pattern=rule.pattern.graph,
            node_constraint=self._lift_constraint_fn(
                fn=rule.node_constraint, impl=impl, pattern=rule.pattern
            ),
        )
        matches = get_rewriteable_matches(
            matches=matches, original=impl.graph, interface_nodes=rule.interface
        )
        self._prepare_contexts(matches)
        rewritten = self._rewrite_raw_graphs()

        new_impl = Implementation(graph=rewritten, data=copy.deepcopy(impl.data))
        new_impl.sync_data_with_graph()

        self._set_new_impl_for_contexts(new_impl)
        self._copy_node_data_from_replacements_to_new_impl()

        return new_impl

    def _set_new_impl_for_contexts(self, new_impl: Implementation) -> None:
        for ctx in self._current_contexts:
            ctx.new_impl = new_impl

    def _prepare_contexts(self, matches: Iterable[dict[str, str]]) -> None:
        for match in matches:
            if self._current_rule is None or self._current_impl is None:
                raise ValueError(
                    "Current rule or implementation is not set. "
                    "This should not happen, please report a bug."
                )
            self._current_contexts.append(
                _RewritingContext(
                    match=match,
                    rule=self._current_rule,
                    original_impl=self._current_impl,
                )
            )

    def _rewrite_raw_graphs(self):
        rewritten = self._current_impl.graph
        for ctx in self._current_contexts:
            rewritten, replacement_map = rewrite(
                replacement=ctx.replacement.graph,
                original=rewritten,
                match=ctx.match,
                lhs={x: x for x in self._current_rule.interface},
                rhs={x: x for x in self._current_rule.interface},
            )
            ctx.replacement_map = replacement_map

        return rewritten

    def _copy_node_data_from_replacements_to_new_impl(
        self,
    ) -> None:
        if self._current_rule is None:
            raise ValueError(
                "Current rule is not set. This should not happen, please report a bug."
            )
        for ctx in self._current_contexts:
            for node_name in ctx.replacement.nodes:
                if node_name not in self._current_rule.interface:
                    node = ctx.replacement.nodes[node_name]
                    ctx.new_impl.nodes[node_name].data.update(node.data)

    def apply(self, impl: Implementation) -> Implementation:
        for rule in self._rules:
            impl = self._apply_rule(impl, rule)
        return impl


class RewriteRule:
    """Specifies how `Rewriter` should create a new `Implementation` from an existing one.

    A rule consists of

    - a pattern to match in the original implementation. Which
    attributes these nodes should have depends on your
    implementation of the `node_constraint` function.
    - a replacement to apply when the pattern is matched. The
    `Rewriter` will use the `replacement` function to create a
    new `Implementation`. The function receives the matched
    part of the implementation as an argument, so you can build
    the replacement based on the parameters found in your
    matched pattern. The node names in this matched
    subimplementation are remapped from the original graph to
    the pattern graph, so you can access the data dictionary
    using the names from the pattern graph. E.g., if the
    pattern specifies a node `'conv'` you can acces the data of
    the original implementation that this node corresponds to
    using `matched.nodes['conv'].data`.
    - a node constraint to check if a pattern node matches a
    node in the original implementation
    - an interface that specifies which nodes are part of the
    pattern and of the replacement, nodes that are part of the
    interface are neither replaced nor initialized.
    Instead they are used to connect the pattern and the
    replacement.
    """

    def __init__(
        self,
        pattern: Implementation,
        replacement: Callable[[RemappedSubImplementation], Implementation],
        node_constraint: Callable[[Node, Node], bool],
        interface: set[str],
    ):
        self.pattern = pattern
        self.replacement = replacement
        self.node_constraint = node_constraint
        self.interface = interface
