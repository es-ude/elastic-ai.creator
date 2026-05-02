from abc import ABC, abstractmethod
from collections.abc import Iterable

from elasticai.creator_plugins.grouped_filter import FilterParameters

from ._ir import (
    DataGraph,
    NameRegistry,
    Node,
    NodeConstraint,
    Registry,
    Rule,
    attribute,
    build_sequential_ir,
    ir_factory,
    pattern_rule,
    sequential_with_interface,
    wrap_graph,
)


class PrecomputationStrategy(ABC):
    def __init__(self):
        self._filter_params = FilterParameters(
            kernel_size=1, in_channels=1, out_channels=1
        )
        self._module: tuple[DataGraph, Registry[DataGraph]] = build_sequential_ir(
            sequence=(), registry=dict()
        )

    @property
    @abstractmethod
    def pattern_graph(self) -> DataGraph: ...

    @abstractmethod
    def constraint(self, registry: Registry, /) -> NodeConstraint: ...

    @abstractmethod
    def get_io_pairs(self) -> Iterable[Iterable[tuple[str, str]]]: ...

    @abstractmethod
    def get_filter_parameters(self) -> FilterParameters: ...

    def set_module(self, g: DataGraph, reg: Registry[DataGraph]) -> None:
        self._module = g, reg

    def _get_impl(self, node: Node | str) -> DataGraph:
        g, reg = self._module
        if isinstance(node, str):
            return reg[g.nodes[node].implementation]
        return reg[node.implementation]


def make_precompute_rule(strategy: PrecomputationStrategy) -> Rule:
    interface = set()
    for n in strategy.pattern_graph.nodes.values():
        if n.type == "interface":
            interface.add(n.name)

    def replacement_fn(
        match: DataGraph, registry: Registry[DataGraph]
    ) -> tuple[DataGraph, Registry[DataGraph]]:
        name_registry = NameRegistry()
        name_registry.prepopulate(registry)
        strategy.set_module(
            wrap_graph(match),
            Registry(**{k: wrap_graph(v) for k, v in registry.items()}),
        )
        lutrons: dict[str, DataGraph] = {}
        for truth_table in strategy.get_io_pairs():
            name = name_registry.get_unique_name("lutron")
            _truth_table = tuple(truth_table)
            lutron = dict(
                truth_table=truth_table,
                type="lutron",
                input_size=len(_truth_table[0][0]),
                output_size=len(_truth_table[0][1]),
            )
            lutrons[name] = ir_factory.graph(attribute(lutron))
        lutron_filter_attributes = attribute(
            type="grouped_filter",
            kernel_per_group=tuple(lutrons),
            **strategy.get_filter_parameters().as_dict(),
        )
        lutron_filter = ir_factory.graph(lutron_filter_attributes)
        lutron_filter_name = name_registry.get_unique_name("lutron_filter")
        registry = registry | lutrons | {lutron_filter_name: lutron_filter}
        replacement = sequential_with_interface(
            ("lutron_filter", "filter"),
        )
        replacement = replacement.add_node(
            "lutron_filter",
            lutron_filter_attributes.drop("kernel_per_group")
            | dict(type="filter", implementation=lutron_filter_name),
        )
        return replacement, registry

    return pattern_rule(
        strategy.pattern_graph,
        replacement_fn=replacement_fn,
        make_node_constraint=strategy.constraint,
        interface=interface,
    )
