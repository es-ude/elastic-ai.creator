from typing import Any

from pytest import fixture

from elasticai.creator.ir import Node
from elasticai.creator.ir.helpers import Shape
from elasticai.creator.ir2vhdl import Implementation, VhdlNode


@fixture
def impl() -> Implementation:
    return Implementation(
        name="conv1",
        type="conv",
        attributes={"a": 1},
        nodes=(Node(dict(name="x", type="y")),),
    )


@fixture
def data() -> dict[str, Any]:
    return {
        "name": "conv1",
        "type": "conv",
        "nodes": [
            {"name": "x", "type": "y"},
        ],
        "edges": [],
        "attributes": {"a": 1},
    }


def test_store_as_dict(data, impl):
    assert data == impl.asdict()


def test_load_from_dict(data):
    assert data == Implementation.fromdict(data).asdict()


def test_can_access_attributes_of_vhdl_node():
    n = VhdlNode({"name": "a", "type": "b", "implementation": "c", "stride": 2})
    n.input_shape = Shape(1)
    n.output_shape = Shape(2)
    assert n.attributes["stride"] == 2