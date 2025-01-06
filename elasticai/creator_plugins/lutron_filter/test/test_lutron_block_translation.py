from torch.nn import BatchNorm1d, Conv1d, Module

from elasticai.creator_plugins.lutron_filter.torch.lutron_modules import LutronConv
from elasticai.creator_plugins.lutron_filter.torch2ir import (
    Torch2IrConverter,
    add_lutron_filter_impl,
)

from ._imports import FilterParameters
from .lutron_block_matcher import LutronBlockMatcher
from .models import (
    HumbleBinarization as BinaryQuantization,
)


def test_produce_no_names_starting_with_underscore():
    class Model(Module):
        def __init__(self):
            super().__init__()
            self._conv = Conv1d(in_channels=1, kernel_size=2, out_channels=1)
            self._bn = BatchNorm1d(1)
            self._bin = BinaryQuantization()

        def forward(self, x):
            return self._bin(self._bn(self._conv(x)))

    m = Model()
    converter = Torch2IrConverter(
        leafs=("HumbleBinarization", "SmoothSigmoid"),
        block_matcher_factory=LutronBlockMatcher,
        type_lists_for_reordering=(
            ("Conv1d", "BatchNorm1d", "HumbleBinarization", "MaxPool1d"),
        ),
    )
    ir = converter.convert(m, input_shape=(2, 1, 4)).values()

    names = set()
    for impl in ir:
        for n in impl.nodes.values():
            names.add(n.name)
            names.add(n.implementation)

    names_with_underscore = [n for n in names if n.startswith("_")]
    assert 0 == len(
        names_with_underscore
    ), f"expected no names with '_', but found {','.join(names_with_underscore)}"


def test_lutron_size_is_computed_correctly():
    """TODO: check why this test is not deterministic"""
    module = LutronConv(
        Conv1d(kernel_size=3, in_channels=6, out_channels=4, groups=2),
        FilterParameters(
            kernel_size=3,
            in_channels=6,
            groups=2,
            out_channels=4,
            input_size=4,
            output_size=2,
        ),
    )
    registry = {}
    lutrons = []
    implementations = tuple(add_lutron_filter_impl("example_filter", module, registry))
    for impl in implementations:
        if impl.type == "lutron":
            lutrons.append(impl)

    expected = [(9, 2), (9, 2)]
    actual = []
    for tron in lutrons:
        actual.append((tron.attributes["input_size"], tron.attributes["output_size"]))

    assert expected == actual
