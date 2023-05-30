from elasticai.creator.hdl.auto_wire_protocols.autowiring import AutoWirer, DataFlowNode


def _wire(top: DataFlowNode, graph: tuple[DataFlowNode, ...]):
    _autowirer = AutoWirer()
    _autowirer.wire(top, graph)
    return _autowirer.connections()


def test_wire_buffered_entity_to_top_module():
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


def test_wire_two_buffered_entities():
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


def test_wire_unbuffered_entity():
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
    """
    just use a list with buffered and a list with mixed entities
    and apply wire defs accordingly
    """
    assert connections == expected_connections
