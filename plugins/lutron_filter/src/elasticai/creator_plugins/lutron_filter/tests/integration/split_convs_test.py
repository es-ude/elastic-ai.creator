import pytest
from elasticai.creator_plugins.lutron_filter.rules.split import make_split_conv_rule
from pytest import Subtests

from elasticai.creator_plugins.grouped_filter import FilterParameters
from elasticai.creator_plugins.lutron_filter.rules import _ir as ir
from elasticai.creator_plugins.lutron_filter.rules._ir import FilterDecorator


def conv1d(
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
    stride: int = 1,
    groups: int = 1,
):
    return ir.attribute(
        type="conv1d",
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        bias=bias,
        stride=stride,
        groups=groups,
    )


def bnorm(num_features: int):
    return ir.attribute(type="batchnorm1d", num_features=num_features)


def binarize():
    return ir.attribute(type="binarize")


def node(name: str, type: str, implementation: str) -> ir.Node:
    return ir.ir_factory.node(
        name,
        ir.attribute(
            type=type,
            implementation=implementation,
        ),
    )


@pytest.fixture
def make_original():
    def _do_make(kernel_size: int, in_channels: int, out_channels):

        return ir.build_sequential_ir(
            sequence=(
                "conv1d_0",
                "batchnorm1d_0",
                "binarize_0",
            ),
            registry=dict(
                conv1d_0=conv1d(
                    kernel_size=kernel_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                ),
                batchnorm1d_0=bnorm(out_channels),
                binarize_0=binarize(),
            ),
        )

    return _do_make


def test_can_replace_conv_with_split(subtests: Subtests, make_original) -> None:
    original_graph, original_reg = make_original(
        kernel_size=2, in_channels=2, out_channels=2
    )

    def replace_filter_params(
        filter: FilterParameters | None = None,
    ) -> tuple[FilterParameters, FilterParameters]:
        return FilterParameters(
            kernel_size=2,
            in_channels=2,
            out_channels=2,
            groups=2,
        ), FilterParameters(kernel_size=1, in_channels=2, out_channels=2, groups=1)

    rule = make_split_conv_rule(replacer=replace_filter_params)

    result_graph, result_reg = rule(original_graph, original_reg)

    with subtests.test("edges connect correct node types"):
        assert get_type_sequence(result_graph) == [
            "input",
            "conv1d",
            "batchnorm1d",
            "binarize",
            "conv1d",
            "batchnorm1d",
            "binarize",
            "output",
        ]
    new_conv_name = get_impl_name_sequence(result_graph)[1]
    expected_param, _ = replace_filter_params()
    for item in ("out_channels", "groups", "in_channels", "kernel_size"):
        with subtests.test(f"expected first conv implementation is correct for {item}"):
            new_conv = FilterDecorator(result_reg[new_conv_name])
            assert getattr(expected_param, item) == getattr(new_conv, item)

    new_conv_name = get_impl_name_sequence(result_graph)[4]
    _, expected_param = replace_filter_params()
    for item in ("out_channels", "groups", "in_channels", "kernel_size"):
        with subtests.test(
            f"expected second conv implementation is correct for {item}"
        ):
            new_conv = FilterDecorator(result_reg[new_conv_name])
            assert getattr(expected_param, item) == getattr(new_conv, item)


@pytest.mark.parametrize(
    ["channels"],
    [(2,), (4,)],
)
def test_uses_channels_from_filter_replacer_function(make_original, channels: int):
    def replace_filter_params(
        filter: FilterParameters,
    ) -> tuple[FilterParameters, FilterParameters]:
        return FilterParameters(
            kernel_size=filter.kernel_size,
            in_channels=channels,
            out_channels=2 * channels,
            groups=2,
        ), FilterParameters(
            kernel_size=1, in_channels=2 * channels, out_channels=channels, groups=1
        )

    rule = make_split_conv_rule(replacer=replace_filter_params)
    _, new_reg = rule(
        *make_original(
            kernel_size=3,
            in_channels=channels,
            out_channels=channels,
        )
    )
    assert new_reg["conv_a"].attributes["out_channels"] == 2 * channels


def test_uses_stride_from_filter_params(make_original):
    def replace_filter_params(
        filter: FilterParameters,
    ) -> tuple[FilterParameters, FilterParameters]:
        return FilterParameters(
            kernel_size=filter.kernel_size,
            in_channels=filter.in_channels,
            out_channels=filter.in_channels,
            groups=1,
        ), FilterParameters(
            kernel_size=1,
            in_channels=filter.in_channels,
            out_channels=filter.out_channels,
            groups=1,
            stride=2,
        )

    rule = make_split_conv_rule(replacer=replace_filter_params)
    _, new_reg = rule(
        *make_original(
            kernel_size=3,
            in_channels=1,
            out_channels=1,
        )
    )
    assert FilterDecorator(new_reg["conv_b"]).stride == 2


def get_edge_sequence(g: ir.DataGraph) -> list[tuple[str, str]]:
    seq: list[tuple[str, str]] = []
    current_nodes = ["input"]
    while len(current_nodes) > 0:
        next_current_nodes = []
        for current_node in current_nodes:
            for n in g.successors[current_node]:
                seq.append((current_node, n))
            next_current_nodes.extend(g.successors[current_node])
        current_nodes = sorted(list(set(next_current_nodes)))
    return seq


def get_node_sequence(g: ir.DataGraph) -> list[str]:
    seq: list[str] = []
    for a, b in get_edge_sequence(g):
        if a not in seq:
            seq.append(a)
        if b not in seq:
            seq.append(b)
    return seq


def get_type_sequence(g: ir.DataGraph) -> list[str]:
    return [g.nodes[a].type for a in get_node_sequence(g)]


def get_impl_name_sequence(g: ir.DataGraph) -> list[str]:
    return [g.nodes[a].implementation for a in get_node_sequence(g)]
