import torch
from torch.fx import GraphModule

from elasticai.creator_plugins.lutron_filter.torch.transformation_utils import (
    CallSequenceRecorder,
    LeafTracer,
    LutronTracer,
)

from .models import HumbleBinarization, Model


def test_recording_call_sequence_with_vanilla_tracer():
    m = Model(2)
    t = LeafTracer((HumbleBinarization,))
    g = t.trace(m)
    gm = GraphModule(m, g)
    gm.eval()
    rec = CallSequenceRecorder(gm)
    rec.run(torch.randn((1, 1, 2)))
    expected = [
        ("block0._wrapped.depthwise.conv", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.mpool", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.bn", torch.Size([1, 1, 2])),
        ("block0._wrapped.depthwise.activation", torch.Size([1, 1, 2])),
        ("block0._wrapped.pointwise.conv", torch.Size([1, 1, 2])),
        ("block0._wrapped.pointwise.mpool", torch.Size([1, 2, 2])),
        ("block0._wrapped.pointwise.bn", torch.Size([1, 2, 2])),
        ("block0._wrapped.pointwise.activation", torch.Size([1, 2, 2])),
        ("block1._wrapped.depthwise.conv", torch.Size([1, 2, 2])),
        ("block0._wrapped.depthwise.mpool", torch.Size([1, 2, 2])),
        ("block1._wrapped.depthwise.bn", torch.Size([1, 2, 2])),
        ("block1._wrapped.depthwise.activation", torch.Size([1, 2, 2])),
        ("block1._wrapped.pointwise.conv", torch.Size([1, 2, 2])),
        ("block0._wrapped.depthwise.mpool", torch.Size([1, 1, 2])),
        ("block1._wrapped.pointwise.bn", torch.Size([1, 1, 2])),
        ("block1._wrapped.pointwise.activation", torch.Size([1, 1, 2])),
    ]
    assert expected == rec.call_sequence


def test_recording_with_lutron_tracer():
    m = Model(2)
    t = LutronTracer()
    g = t.trace(m)
    gm = GraphModule(m, g)
    rec = CallSequenceRecorder(gm)
    rec.run(torch.randn((1, 1, 2)))
    expected = [
        ("block0", torch.Size([1, 1, 2])),
        ("block1", torch.Size([1, 2, 2])),
    ]
    assert expected == rec.call_sequence
