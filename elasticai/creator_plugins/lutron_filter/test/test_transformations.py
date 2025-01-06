import torch
from torch.fx import Tracer
from torch.nn import BatchNorm1d, Conv1d, Identity, Linear, MaxPool1d

from elasticai.creator_plugins.lutron_filter.torch.transformation_utils import (
    CallSequenceRecorder,
    LeafTracer,
)
from elasticai.creator_plugins.lutron_filter.torch.transformations import (
    remove_identities,
)
from elasticai.creator_plugins.lutron_filter.torch.transformations import (
    reorder_conv_blocks as _reorder_conv_blocks,
)

from .models import (
    HumbleBinarization as BinaryQuantization,
)
from .models import (
    Model,
    ModelWithBinaryQuantization,
)


def reorder_conv_blocks(m, g):
    type_lists = [
        (Conv1d, MaxPool1d, BatchNorm1d, Identity),
        (Conv1d, Identity, Identity, Identity),
        (Conv1d, Identity, BatchNorm1d, Identity),
        (Conv1d, MaxPool1d, BatchNorm1d, BinaryQuantization),
        (Conv1d, Identity, BatchNorm1d, BinaryQuantization),
    ]
    return _reorder_conv_blocks(m, g, type_lists=type_lists)


def test_reordering_conv_blocks():
    m = Model(1)
    expected = [
        ("block0._wrapped.depthwise.conv", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.bn", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.activation", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.mpool", torch.Size([1, 1, 2])),
        ("block0._wrapped.pointwise.conv", torch.Size([1, 1, 2])),
        ("block0._wrapped.pointwise.bn", torch.Size([1, 2, 2])),
        ("block0._wrapped.pointwise.activation", torch.Size([1, 2, 2])),
        ("block0._wrapped.pointwise.mpool", torch.Size([1, 2, 2])),
    ]
    t = LeafTracer((BinaryQuantization,))
    g = t.trace(m)
    gm = reorder_conv_blocks(m, g)
    rec = CallSequenceRecorder(gm)
    rec.run(torch.randn((1, 1, 2)))
    assert expected == rec.call_sequence


def test_removing_identities():
    m = Model(1)
    expected = [
        "block0._wrapped.depthwise.conv",
        "block0._wrapped.depthwise.bn",
        "block0._wrapped.pointwise.conv",
        "block0._wrapped.pointwise.mpool",
        "block0._wrapped.pointwise.bn",
    ]
    t = Tracer()
    g = t.trace(m)
    gm = remove_identities(m, g)

    rec = CallSequenceRecorder(gm)
    rec.run(torch.randn((1, 1, 2)))
    actual = list(x[0] for x in rec.call_sequence)
    assert expected == actual


def test_reordering_conv_blocks_with_quant_act():
    m = ModelWithBinaryQuantization(1)
    expected = [
        ("block0._wrapped.depthwise.conv", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.bn", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.activation", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.mpool", torch.Size([1, 1, 2])),
        ("block0._wrapped.pointwise.conv", torch.Size([1, 1, 2])),
        ("block0._wrapped.pointwise.bn", torch.Size([1, 2, 2])),
        ("block0._wrapped.pointwise.activation", torch.Size([1, 2, 2])),
        ("block0._wrapped.pointwise.mpool", torch.Size([1, 2, 2])),
    ]

    g = LeafTracer((BinaryQuantization,)).trace(m)
    gm = reorder_conv_blocks(m, g)
    rec = CallSequenceRecorder(gm)
    rec.run(torch.randn((1, 1, 2)))
    assert expected == rec.call_sequence


def get_call_sequence(gm, input_shape):
    rec = CallSequenceRecorder(gm)
    rec.run(torch.randn(input_shape))
    return tuple(x[0] for x in rec.call_sequence)


def test_reordering_sequential_with_bin_activation():
    m = torch.nn.Sequential(
        Conv1d(in_channels=1, out_channels=1, kernel_size=1),
        MaxPool1d(kernel_size=1),
        BatchNorm1d(num_features=1),
        BinaryQuantization(),
        Conv1d(in_channels=1, out_channels=1, kernel_size=1),
    )
    m.eval()
    g = LeafTracer((BinaryQuantization,)).trace(m)
    gm = reorder_conv_blocks(m, g)
    actual = get_call_sequence(gm, (1, 1, 1))
    expected = ("0", "2", "3", "1", "4")
    assert expected == actual


def test_reorder_block_with_bin_activation_produces_correct_code():
    m = torch.nn.Sequential(
        Conv1d(in_channels=1, out_channels=1, kernel_size=1),
        MaxPool1d(kernel_size=1),
        BatchNorm1d(num_features=1),
        BinaryQuantization(),
        Linear(in_features=1, out_features=1),
    )
    m.eval()
    g = LeafTracer((BinaryQuantization,)).trace(m)
    gm = reorder_conv_blocks(m, g)
    expected = """


def forward(self, input):
    input_1 = input
    _0 = getattr(self, "0")(input_1);  input_1 = None
    _2 = getattr(self, "2")(_0);  _0 = None
    _3 = getattr(self, "3")(_2);  _2 = None
    _1 = getattr(self, "1")(_3);  _3 = None
    _4 = getattr(self, "4")(_1);  _1 = None
    return _4
    """
    assert expected == str(gm.code)
