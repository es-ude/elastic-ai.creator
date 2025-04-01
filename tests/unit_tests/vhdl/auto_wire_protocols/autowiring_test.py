import re

import pytest

from elasticai.creator.vhdl.auto_wire_protocols.autowiring import (
    AutoWirer,
    AutoWiringProtocolViolation,
    DataFlowNode,
)


def _wire(
    top: DataFlowNode, graph: tuple[DataFlowNode, ...]
) -> dict[tuple[str, str], tuple[str, str]]:
    _autowirer = AutoWirer()
    _autowirer.wire(top, graph)
    return _autowirer.connections()


def _build_expected_connections(spec: str):
    regex = r"\s*(\w+)\s*,\s*(\w+)\s*:\s*(\w+)\s*,\s*(\w+)\s*\n*"
    result = {}
    for match in re.finditer(regex, spec):
        dst_node, dst_signal, source_node, source_signal = match.groups()
        result[(dst_node, dst_signal)] = (source_node, source_signal)
    return result


def test_wire_buffered_entity_to_top_module() -> None:
    child = DataFlowNode.buffered("child")
    connections = _wire(top=DataFlowNode.top("top"), graph=(child,))
    expected_connections = {
        ("child", "x"): ("top", "x"),
        ("child", "y_address"): ("top", "y_address"),
        ("child", "clock"): ("top", "clock"),
        ("child", "enable"): ("top", "enable"),
        ("top", "y"): ("child", "y"),
        ("top", "x_address"): ("child", "x_address"),
        ("top", "done"): ("child", "done"),
    }
    assert connections == expected_connections


def test_wire_two_buffered_entities() -> None:
    a = DataFlowNode.buffered("a")
    b = DataFlowNode.buffered("b")
    top = DataFlowNode.top("top")
    expected_connections = {
        ("a", "clock"): ("top", "clock"),
        ("a", "x"): ("top", "x"),
        ("a", "enable"): ("top", "enable"),
        ("top", "x_address"): ("a", "x_address"),
        ("a", "y_address"): ("b", "x_address"),
        ("b", "clock"): ("top", "clock"),
        ("b", "x"): ("a", "y"),
        ("b", "y_address"): ("top", "y_address"),
        ("b", "enable"): ("a", "done"),
        ("top", "y"): ("b", "y"),
        ("top", "done"): ("b", "done"),
    }
    connections = _wire(top=top, graph=(a, b))
    assert connections == expected_connections


def test_wire_unbuffered_entity() -> None:
    a = DataFlowNode.unbuffered("a")
    expected_connections = {
        ("top", "y"): ("a", "y"),
        ("top", "done"): ("top", "enable"),
        ("top", "x_address"): ("top", "y_address"),
        ("a", "enable"): ("top", "enable"),
        ("a", "x"): ("top", "x"),
        ("a", "clock"): ("top", "clock"),
    }
    connections = _wire(top=DataFlowNode.top("top"), graph=(a,))
    assert connections == expected_connections


def test_wire_buffered_and_unbuffered() -> None:
    a = DataFlowNode.buffered("a")
    b = DataFlowNode.unbuffered("b")
    top = DataFlowNode.top("top")
    spec = """
        top, x_address : a, x_address
        a, clock : top, clock
        a, enable : top, enable
        a, y_address : top, y_address
        a, x : top, x
        b, enable : a, done
        b, clock : top, clock
        b, x : a, y
        top, y: b, y
        top, done: a, done
        """
    expected_connections = _build_expected_connections(spec)
    connections = _wire(top=top, graph=(a, b))
    assert connections == expected_connections


def test_wiring_unknown_signals_yields_error() -> None:
    a = DataFlowNode(name="a", dsts=("signal_a",), sources=tuple())
    with pytest.raises(AutoWiringProtocolViolation):
        _wire(top=DataFlowNode.top("top"), graph=(a,))
