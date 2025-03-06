from collections.abc import Sequence

from elasticai.creator import graph as gr
from elasticai.creator import ir_transforms as ir_t
from elasticai.creator import torch2ir as t2ir


def add_sequence(
    impl: t2ir.Implementation, sequence: Sequence[tuple[str, str]]
) -> None:
    for name, _type in sequence:
        impl.add_node(name=name, data=dict(type=_type))
    for (src, _), (dst, _) in zip(sequence, sequence[1:]):
        impl.add_edge(src=src, dst=dst, data={})


def test_rebuild_new_implementation():
    impl = t2ir.Implementation(
        graph=gr.BaseGraph(),
        name="test",
        type="test",
    )
    add_sequence(impl, [("input", "input"), ("conv", "conv1d"), ("bin", "binarize")])
    pattern = ir_t.build_sequential_pattern(
        ("start", {"input"}),
        ("filter", {"conv1d"}),
        ("end", {"binarize"}),
    )
    new_graphs = ir_t.move_pattern_to_subimpls(
        original=impl,
        pattern=pattern,
        basename="lutron_module",
        replacement_data_fn=lambda _: {
            "lutron_module": {
                "type": "lutron_module",
                "random_data": "random",
            },
        },
        extracted_data_fn=lambda _: {"filter": {"more_data": "more"}},
    )
    assert new_graphs[0].data == {
        "nodes": {
            "input": {},
            "filter": {"type": "conv1d", "more_data": "more"},
            "output": {},
        },
        "edges": {"start": {"filter": {}}, "filter": {"end": {}}},
        "name": "test_lutron_module",
        "type": "lutron_module",
    }
    assert new_graphs[1].data == {
        "nodes": {
            "lutron_module": {
                "type": "lutron_module",
                "random_data": "random",
                "implementation": "test_lutron_module",
            },
            "bin": {"type": "binarize"},
            "input": {"type": "input"},
        },
        "edges": {},
    }
