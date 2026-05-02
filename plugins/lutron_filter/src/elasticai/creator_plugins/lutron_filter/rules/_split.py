from collections.abc import Callable, Iterator

from elasticai.creator.ir import AttributeMapping, attribute
from elasticai.creator_plugins.grouped_filter import FilterParameters

from ._ir import (
    DataGraph,
    FilterDecorator,
    NameRegistry,
    Node,
    NodeConstraint,
    Registry,
    Rule,
    node,
    pattern_rule,
    sequential_with_interface,
)

FilterParamsProducer = Callable[[FilterParameters], tuple[FilterParameters, ...]]


def make_split_conv_rule(
    replacer: FilterParamsProducer,
    additional_node_constraint: Callable[[Node, Node], bool] = lambda _, __: True,
) -> Rule[DataGraph, DataGraph]:

    return pattern_rule(
        graph=sequential_with_interface("conv1d"),
        replacement_fn=_ReplacementFN(replacer),
        make_node_constraint=_make_make_node_constraint(additional_node_constraint),
    )


def _make_make_node_constraint(
    additional_constraint: Callable[[Node, Node], bool],
) -> Callable[[Registry[DataGraph]], NodeConstraint]:
    def make_node_constraint(reg: Registry[DataGraph]) -> NodeConstraint:
        def node_constraint(pattern_node: Node, graph_node: Node, /) -> bool:
            if pattern_node.type == "interface":
                return True
            if graph_node.type == pattern_node.type:
                filter_impl = FilterDecorator(reg[graph_node.implementation])
                return filter_impl.groups == 1 and filter_impl.kernel_size > 1
            return False

        def combined_node_constraint(pattern_node: Node, graph_node: Node, /) -> bool:
            return node_constraint(pattern_node, graph_node) and additional_constraint(
                pattern_node, graph_node
            )

        return node_constraint

    return make_node_constraint


class _ReplacementFN:
    def __init__(self, filter_replacer: FilterParamsProducer) -> None:
        self._filter_replacer = filter_replacer
        self._suffixes = "abcdefghijklmnopqrstuvwxyz"
        self._node_seq_filter_pairs = None

    def __call__(
        self, match: DataGraph, reg: Registry[DataGraph]
    ) -> tuple[DataGraph, Registry[DataGraph]]:
        self._match = match
        self._reg = reg
        self._new_filters = self._get_new_filter_pair()
        if len(self._new_filters) > len(self._suffixes):
            raise ValueError(
                f"too many filters provided. Got: {len(self._new_filters)}, expected: at most {len(self._suffixes)}"
            )
        self._update_name_registry()
        self._update_replacement_graph_and_registry()

        return self._replacement, self._reg

    def _update_name_registry(self) -> None:
        self._naming = NameRegistry()
        self._naming.prepopulate(self._reg.keys())

    def _new_name(self, name: str) -> str:
        return self._naming.get_unique_name(name)

    def _update_replacement_graph_and_registry(self) -> None:
        node_sequence = [node("start", "interface")]

        for nodes, filter in self._get_node_sequences_and_filters():
            node_sequence.extend(nodes)
            for n, fn in zip(
                nodes,
                (self._add_conv_to_reg, self._add_bnorm_to_reg, self._add_bin_to_reg),
            ):
                fn(n.implementation, filter)

        node_sequence.append(node("end", "interface"))
        edges = [
            (src.name, dst.name)
            for src, dst in zip(node_sequence[:-1], node_sequence[1:])
        ]
        replacement = self._match.clear().add_edges(*edges).add_nodes(*node_sequence)
        self._replacement = replacement

    def _get_impl_for_matched(self, node: str) -> DataGraph:
        return self._reg[self._match.nodes[node].implementation]

    def _get_impl_for_replacement(self, node: str) -> DataGraph:
        return self._reg[self._replacement.nodes[node].implementation]

    def _get_filter_impl(self, node: str) -> FilterDecorator[DataGraph]:
        return FilterDecorator(self._reg[self._match.nodes[node].implementation])

    def _get_new_filter_pair(self) -> tuple[FilterParameters, ...]:
        return self._filter_replacer(self._get_filter_impl("conv1d").filter_parameters)

    def _build_node_sequence(self, suffix: str) -> list[Node]:
        conv_name = f"conv_{suffix}"
        bnorm_name = f"bnorm_{suffix}"
        bin_name = f"bin_{suffix}"
        return [
            node(
                conv_name,
                type="conv1d",
                implementation=self._new_name(conv_name),
            ),
            node(bnorm_name, "batchnorm1d", self._new_name(bnorm_name)),
            node(bin_name, "binarize", self._new_name(bin_name)),
        ]

    def _get_node_sequences_and_filters(
        self,
    ) -> Iterator[tuple[list[Node], FilterParameters]]:
        if self._node_seq_filter_pairs is None:
            self._node_seq_filter_pairs = []
            for filter_id, (suffix, filter) in enumerate(
                zip(self._suffixes, self._new_filters)
            ):
                nodes = self._build_node_sequence(suffix)
                if filter_id == len(self._new_filters) - 1:
                    nodes = [nodes[0]]
                yield nodes, filter
                self._node_seq_filter_pairs.append((nodes, filter))
        else:
            yield from self._node_seq_filter_pairs

    def _add_to_reg(self, name: str, type: str, attributes: AttributeMapping):
        self._reg = self._reg | {
            name: self._match.clear().with_attributes(
                attribute(
                    type=type,
                )
                | attributes
            )
        }

    def _add_conv_to_reg(self, name: str, f: FilterParameters) -> None:
        self._add_to_reg(
            name,
            type="conv1d",
            attributes=attribute(
                **{
                    k: v
                    for k, v in f.as_dict().items()
                    if k not in ("input_size", "output_size")
                }
            ),
        )

    def _add_bnorm_to_reg(self, name: str, f: FilterParameters) -> None:
        self._add_to_reg(
            name, type="batchnorm1d", attributes=attribute(num_features=f.out_channels)
        )

    def _add_bin_to_reg(self, name: str, _: FilterParameters) -> None:
        self._add_to_reg(name, "binarization", attribute())
