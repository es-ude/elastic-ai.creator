import re
from abc import abstractmethod
from collections.abc import Callable, Collection
from typing import Protocol

from elasticai.creator.graph import get_rewriteable_matches, find_all_subgraphs
from elasticai.creator.ir.attribute import AttributeMapping
from elasticai.creator.ir.datagraph import DataGraph as _DGraph
from elasticai.creator.ir.datagraph import Edge, Node
from elasticai.creator.ir.datagraph_impl import DataGraphImpl, DefaultNodeEdgeFactory
from elasticai.creator.ir.graph import GraphImpl
from elasticai.creator.ir.registry import Registry as _Registry

type DataGraph = _DGraph[
    Node, Edge
]  # bind to upper bound until we get default type args
type Registry = _Registry[
    DataGraph
]  # bind to upper bound until we get default type args
type Rule = Callable[
    [DataGraph, Registry],
    tuple[DataGraph, Registry],
]


class Pattern(Protocol):
    @property
    @abstractmethod
    def graph(self) -> DataGraph: ...

    @property
    def interface(self) -> Collection[str]: ...

    @abstractmethod
    def match(self, g: DataGraph, registry: Registry, /) -> list[dict[str, str]]: ...


class StdPattern(Pattern):
    """Use a simple constraint over `Node`s to find all matching subgraphs.

    This pattern ignores the registry, so you cannot
    analyse attributes of other graphs that nodes might
    refer to.
    Thus, all information your pattern matching needs,
    needs to be present in the graph under match already.
    As this is often the case, the pattern is provided as
    `StdPattern`.

    If you need more control, implement your own pattern.
    """

    def __init__(
        self,
        graph: DataGraph,
        node_constraint: Callable[[Node, Node], bool],
        interface: Collection[str],
    ) -> None:
        self._graph = graph
        self._interface = interface
        self._constraint = node_constraint

    @property
    def graph(self) -> DataGraph:
        return self._graph

    @property
    def interface(self) -> Collection[str]:
        return self._interface

    def match(self, g: DataGraph, registry: Registry, /) -> list[dict[str, str]]:
        def constraint(pattern_node: str, graph_node: str) -> bool:
            return self._constraint(self.graph.nodes[pattern_node], g.nodes[graph_node])

        return find_all_subgraphs(
            pattern=self.graph, graph=g, node_constraint=constraint
        )


class PatternRuleSpec:
    """Specifies how `PatternRule` should create a new `DataGraph` from an existing one.

    It consists of

    - a pattern to match in the original DataGraph.
    - a replacement to apply when the pattern is matched. The
    `PatternRule` will use the `replacement_fn` function to create a
    new `DataGraph`. The function receives the matched
    part of the DataGraph as an argument, so you can build
    the replacement based on the parameters found in your
    matched pattern. The node names in this matched
    subimplementation are remapped from the original graph to
    the pattern graph, so you can access the data dictionary
    using the names from the pattern graph. E.g., if the
    pattern specifies a node `'conv'` you can acces the data of
    the original DataGraph that this node corresponds to
    using `matched.nodes['conv'].data`.
    - an interface that specifies which nodes are part of the
    pattern and of the replacement, nodes that are part of the
    interface are neither replaced nor initialized.
    Instead they are used to connect the pattern and the
    replacement.
    """

    def __init__(
        self,
        pattern: Pattern,
        replacement_fn: Rule,
    ) -> None:
        self.pattern = pattern
        self.replacement_fn = replacement_fn

    @property
    def interface(self) -> Collection[str]:
        return self.pattern.interface

    def match(self, g: DataGraph, registry: Registry) -> list[dict[str, str]]:
        return self.pattern.match(g, registry)

    def create_replacement(
        self, g: DataGraph, registry: Registry
    ) -> tuple[DataGraph, Registry]:
        return self.replacement_fn(g, registry)


class PatternRule:
    def __init__(self, spec: PatternRuleSpec):
        self._spec = spec

    def _validate_interface_compatibility(
        self, nodes: Collection[str], graph_type: str
    ) -> None:
        missing_nodes = set()
        for node in self._spec.interface:
            if node not in nodes:
                missing_nodes.add(node)
        if len(missing_nodes) > 0:
            raise ValueError(
                f"{graph_type} is missing interface nodes: {missing_nodes}"
            )

    def _validate_replacement_interface_compatibility(self, nodes) -> None:
        self._validate_interface_compatibility(nodes, "Replacement")

    def _validate_pattern_interface_compatibility(self) -> None:
        self._validate_interface_compatibility(
            self._spec.pattern.graph.nodes, "Pattern Graph"
        )

    def __call__(
        self, graph: DataGraph, registry: Registry
    ) -> tuple[DataGraph, Registry]:
        self._validate_pattern_interface_compatibility()
        matches = self._spec.pattern.match(graph, registry)
        matches = list(
            get_rewriteable_matches(
                original=graph, matches=matches, interface_nodes=self._spec.interface
            )
        )
        for match in matches:
            graph_mapped_to_matched_pattern = _create_remapped_graph(graph, match)
            repl_graph, registry = self._spec.create_replacement(
                graph_mapped_to_matched_pattern, registry
            )
            self._validate_replacement_interface_compatibility(repl_graph.nodes)
            graph = self._replace_match(graph, repl_graph, match, self._spec.interface)
        return graph, registry

    def _replace_match(
        self,
        original: DataGraph,
        replacement: DataGraph,
        match: dict[str, str],
        interface: Collection[str],
    ) -> DataGraph:
        nodes_to_keep: set[str] = set()
        pattern_to_orig = match
        orig_to_pattern = {k: v for v, k in pattern_to_orig.items()}

        for node in original.successors:
            if node not in orig_to_pattern or orig_to_pattern[node] in interface:
                nodes_to_keep.add(node)

        nodes_to_add: set[str] = set()
        for node in replacement.successors:
            if node not in interface:
                nodes_to_add.add(node)

        edges_to_keep: list[tuple[str, str, AttributeMapping]] = []
        for src in nodes_to_keep:
            for dst in nodes_to_keep:
                if dst in original.successors[src]:
                    edges_to_keep.append((src, dst, original.successors[src][dst]))

        name_generator = _NameGenerator(_NameRegistry().prepopulate(nodes_to_keep))
        old_to_new_replacement_names = {}
        for node in nodes_to_add:
            old_to_new_replacement_names[node] = name_generator.get_name(node)

        name_mapping_for_replacement = {
            v: k for k, v in old_to_new_replacement_names.items()
        } | {pattern_to_orig[n]: n for n in interface}
        new_graph = _create_remapped_graph(
            replacement,
            name_mapping_for_replacement,
        )

        new_graph = new_graph.add_nodes(
            *((node, original.node_attributes[node]) for node in nodes_to_keep)
        ).add_edges(*edges_to_keep)
        return new_graph


def _create_remapped_graph(original: DataGraph, mapping: dict[str, str]) -> DataGraph:
    orig_to_new = {v: k for k, v in mapping.items()}

    remapped_node_attribute = AttributeMapping(
        **{
            orig_to_new[k]: v
            for k, v in original.node_attributes.items()
            if k in orig_to_new
        }
    )
    edges_in_match: set[tuple[str, str]] = set()
    for src in orig_to_new:
        for dst in orig_to_new:
            if (src, dst) in original.edges:
                edges_in_match.add((src, dst))
    remapped_edges: list[tuple[str, str, AttributeMapping]] = []
    for src, dst in edges_in_match:
        remapped_edges.append(
            (orig_to_new[src], orig_to_new[dst], original.successors[src][dst])
        )
    remapped_dgraph = DataGraphImpl(
        factory=DefaultNodeEdgeFactory(),
        attributes=original.attributes,
        graph=GraphImpl(lambda: AttributeMapping()),
        node_attributes=remapped_node_attribute,
    )
    remapped_dgraph = remapped_dgraph.add_edges(*remapped_edges)
    return remapped_dgraph


class _NameRegistry:
    def __init__(self):
        self._registry = {}

    def _get_name_count(self, name):
        return self._registry.get(name, 0)

    def prepopulate(self, names):
        for name in names:
            match = re.match(r"(.+)_(\d+)$", name)
            suffix = 0
            if match:
                name = match.group(1)
                suffix = int(match.group(2))
            suffix = max(suffix, self._get_name_count(name))
            self._registry[name] = suffix
        return self

    def get_unique_name(self, name):
        if name not in self._registry:
            self._registry[name] = 0
            return name

        new_name = f"{name}_{self._registry[name] + 1}"
        self._registry[name] += 1
        return new_name


class _NameGenerator:
    def __init__(self, registry: _NameRegistry) -> None:
        self._registry = registry

    def get_name(self, name: str) -> str:
        new_name = self._registry.get_unique_name(name)
        return new_name
