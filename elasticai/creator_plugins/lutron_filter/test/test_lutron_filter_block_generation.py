from modulefinder import Module
from typing import Iterator

import pytest
import torch.nn
from torch.fx import Graph, GraphModule, Tracer
from torch.nn import BatchNorm1d, Conv1d, Flatten, Linear, PReLU

from elasticai.creator.ir.helpers import FilterParameters
from elasticai.creator_plugins.lutron_filter.torch import (
    LutronModule,
    PatternMatch,
    detect_type_sequences,
)
from elasticai.creator_plugins.lutron_filter.torch.lutron_block_autogen import (
    autogenerate_lutron_blocks,
)
from elasticai.creator_plugins.lutron_filter.torch.transformation_utils import (
    LeafTracer,
    LutronTracer,
    get_module_for_node,
)

from .lutron_block_matcher import LutronBlockMatcher
from .models import (
    HumbleBinarization as BinaryQuantization,
)
from .models import (
    MaxPoolModelForLutronBlocks as Model,
)
from .models import (
    SmoothSigmoid,
    lutron_dw_conv_block,
)


@pytest.fixture
def model() -> Module:
    return Model()


@pytest.fixture
def tracer() -> Tracer:
    return LeafTracer((BinaryQuantization, SmoothSigmoid))


@pytest.fixture
def graph(model, tracer) -> Graph:
    return tracer.trace(model)


@pytest.fixture
def lutron_block_model(model, graph) -> GraphModule:
    m = model
    m_with_lutron_blocks = autogenerate_lutron_blocks(
        m, input_shape=(2, 1, 4), graph=graph, block_matcher_factory=LutronBlockMatcher
    )
    return m_with_lutron_blocks


def test_can_generate_first_mpooling_block(model, lutron_modules_with_mpooling) -> None:
    first_mpool_lutron = lutron_modules_with_mpooling[0]
    first_mpool = tuple(first_mpool_lutron.children())[0]
    assert model.mpool == first_mpool.mpool


@pytest.fixture
def lutron_modules_with_mpooling(lutron_block_model) -> list[LutronModule]:
    layers = []
    for name, module in lutron_block_model.named_children():
        if name.endswith("lutron_filter") and name.startswith("mpool"):
            layers.append(module)
    return layers


@pytest.fixture
def lutron_modules(lutron_block_model) -> list[LutronModule]:
    g = LutronTracer().trace(lutron_block_model)
    layers = []
    for n in g.nodes:
        if n.op == "call_module":
            module = get_module_for_node(n, lutron_block_model)
            if isinstance(module, LutronModule):
                layers.append(module)
    return layers


def test_can_generate_conv_block(lutron_block_model):
    convs = []
    for name, module in lutron_block_model.named_children():
        if name.endswith("lutron_filter") and name.startswith("conv"):
            convs.append(module)
    expected = FilterParameters(
        in_channels=1,
        out_channels=2,
        groups=1,
        kernel_size=1,
        stride=1,
        input_size=2,
        output_size=2,
    )
    assert expected == convs[0].filter_parameters


def test_can_generate_second_mpooling_block(lutron_modules_with_mpooling) -> None:
    expected = FilterParameters(
        in_channels=2, out_channels=2, groups=2, kernel_size=2, stride=2, input_size=2
    )
    assert expected == lutron_modules_with_mpooling[1].filter_parameters


def test_can_infer_shape_of_mpooling_block(lutron_modules_with_mpooling) -> None:
    expected = FilterParameters(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        groups=1,
        stride=2,
        input_size=4,
        output_size=2,
    )
    assert expected == lutron_modules_with_mpooling[0].filter_parameters


def test_can_generate_block_for_linear_layers(tracer):
    m = torch.nn.Sequential(
        Flatten(), Linear(in_features=4, out_features=3), BinaryQuantization()
    )
    expected = FilterParameters(
        in_channels=2,
        out_channels=3,
        kernel_size=2,
        groups=1,
        stride=1,
        input_size=2,
        output_size=1,
    )
    graph = tracer.trace(m)
    m = autogenerate_lutron_blocks(
        m, input_shape=(1, 2, 2), graph=graph, block_matcher_factory=LutronBlockMatcher
    )
    assert expected == tuple(m.children())[0].filter_parameters


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


class LutronBlockMatcherWithDoubleMatches(LutronBlockMatcher):
    def conv1d(self: LutronBlockMatcher) -> Iterator[PatternMatch]:
        type_lists = (
            (Conv1d, BatchNorm1d, PReLU, BinaryQuantization),
            (Conv1d, BatchNorm1d, BinaryQuantization),
            (Conv1d, PReLU, BinaryQuantization),
            (Conv1d, BinaryQuantization),
            (Conv1d, BinaryQuantization),
        )
        patterns = tuple(
            LutronBlockMatcher._pattern_starting_with_node_of_interest(*seq)
            for seq in type_lists
        )
        yield from detect_type_sequences(self._m, self._g, patterns)


def test_ignore_second_match(model, graph):
    m = model
    lutron_block_model = autogenerate_lutron_blocks(
        m,
        input_shape=(2, 1, 4),
        graph=graph,
        block_matcher_factory=LutronBlockMatcherWithDoubleMatches,
    )
    convs = []
    for name, module in lutron_block_model.named_children():
        if name.endswith("lutron_filter") and name.startswith("conv"):
            convs.append(module)
    expected = FilterParameters(
        in_channels=1,
        out_channels=2,
        groups=1,
        kernel_size=1,
        stride=1,
        input_size=2,
        output_size=2,
    )
    assert expected == convs[0].filter_parameters
