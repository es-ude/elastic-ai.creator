from collections import OrderedDict

import pytest
import torch.nn
from torch.nn import BatchNorm1d, Conv1d, Linear, MaxPool1d, Sigmoid

from elasticai.creator_plugins.lutron_filter.torch import detect_type_sequences
from elasticai.creator_plugins.lutron_filter.torch.transformation_utils import (
    LeafTracer,
)

from .lutron_block_matcher import LutronBlockMatcher
from .models import (
    HumbleBinarization as BinaryQuantization,
)
from .models import (
    MaxPoolModelForLutronBlocks as MaxPoolModel,
)
from .models import (
    SmoothSigmoid,
    lutron_dw_conv_block,
)


@pytest.fixture
def binary_quant_matcher() -> LutronBlockMatcher:
    t = LeafTracer((BinaryQuantization,))
    m = MaxPoolModel()
    g = t.trace(m)
    matcher = LutronBlockMatcher(m, g)
    return matcher


def test_detect_precomputable_maxpooling(binary_quant_matcher: LutronBlockMatcher):
    expected = (("mpool",), ("mpool_1",))
    assert expected == tuple(
        tuple(n.name for n in block.matched_sequence)
        for block in binary_quant_matcher.maxpool1d()
    )


def detect_conv_blocks(m, g):
    type_lists = ((Conv1d, BatchNorm1d, BinaryQuantization),)
    patterns = tuple(
        LutronBlockMatcher._pattern_starting_with_node_of_interest(*seq)
        for seq in type_lists
    )
    yield from detect_type_sequences(m, g, patterns)


def test_detect_simple_precomputable_conv_block():
    t = LeafTracer((BinaryQuantization,))
    m = torch.nn.Sequential(
        OrderedDict(
            a=Conv1d(kernel_size=1, in_channels=1, out_channels=1),
            b=BatchNorm1d(num_features=1),
            c=BinaryQuantization(),
            d=MaxPool1d(2),
        )
    )
    m.eval()
    g = t.trace(m)
    expected = (("a", "b", "c"),)
    #
    # class LutronBlockMatcher:
    #     def __init__(self, m, g):
    #         self.m = m
    #         self.g = g
    #
    #     def conv1d(self):
    #         m = self.m
    #         g = self.g
    #         yield from detect_type_sequences(
    #             m,
    #             g,
    #             (
    #                 SequentialPattern(
    #                     nodes_of_interest=(0,),
    #                     type_sequence=(Conv1d, BatchNorm1d, BinaryQuantization),
    #                 ),
    #             ),
    #         )

    matcher = LutronBlockMatcher(m, g)
    assert expected == tuple(
        tuple(n.name for n in block.matched_sequence) for block in matcher.conv1d()
    )


def test_detect_precomputable_conv_block(binary_quant_matcher: LutronBlockMatcher):
    expected = (("conv", "bn", "relu", "quant_1"),)
    assert expected == tuple(
        tuple(n.name for n in block.matched_sequence)
        for block in binary_quant_matcher.conv1d()
    )


class NestedModuleWithSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = lutron_dw_conv_block(in_channels=1, out_channels=1, kernel_size=1)

        self.lin = Linear(in_features=1, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return torch.flatten(
            self.sigmoid(self.lin(torch.flatten(self.block(x), start_dim=1)) / 5.0),
            start_dim=1,
        )


@pytest.fixture
def matcher_for_nested_module():
    t = LeafTracer((BinaryQuantization,))
    model = torch.nn.Sequential(NestedModuleWithSigmoid(), BinaryQuantization())
    g = t.trace(model)
    return LutronBlockMatcher(model, g)


def test_can_detect_linear_precomp_block_when_appending_quant_to_other_act_sequential(
    matcher_for_nested_module: LutronBlockMatcher,
):
    m = matcher_for_nested_module
    expected = (("flatten", "_0_lin", "truediv", "_0_sigmoid", "flatten_1", "_1"),)
    assert expected == tuple(
        tuple(n.name for n in block.matched_sequence) for block in m.linear()
    )


class NestedModuleWithSmoothSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = lutron_dw_conv_block(in_channels=1, out_channels=1, kernel_size=1)

        self.lin = Linear(in_features=1, out_features=1)
        self.sigmoid = SmoothSigmoid()

    def forward(self, x):
        return torch.flatten(
            self.sigmoid(self.lin(torch.flatten(self.block(x), start_dim=1))),
            start_dim=1,
        )


def test_can_detect_linear_precomp_block_with_smooth_sigmoid():
    t = LeafTracer((BinaryQuantization, SmoothSigmoid))
    model = torch.nn.Sequential(NestedModuleWithSmoothSigmoid(), BinaryQuantization())
    g = t.trace(model)
    matcher = LutronBlockMatcher(model, g)
    expected = (("flatten", "_0_lin", "_0_sigmoid", "flatten_1", "_1"),)
    assert expected == tuple(
        tuple(n.name for n in block.matched_sequence) for block in matcher.linear()
    )
