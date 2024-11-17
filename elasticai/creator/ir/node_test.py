from typing import overload

import pytest

from elasticai.creator.ir.abstract_ir_data.mandatory_field import MandatoryField

from .abstract_ir_data import AbstractIRData
from .abstract_ir_data.mandatory_field import TransformableMandatoryField
from .node import Node
from .node import Node as OtherNode


def test_can_create_new_node():
    n = Node.new(name="my_name", type="my_type")
    assert n.name == "my_name"


def test_creating_new_node_without_filling_all_args_yields_error():
    with pytest.raises(ValueError):
        Node.new(name="my_name")


def test_filling_name_arg_twice_leads_to_error():
    with pytest.raises(ValueError):
        Node.new("my_name", "my_type", name="other_name")


def test_attributes_are_merged_into_data():
    n = Node.new(name="my_name", type="my_type", attributes=dict(input_shape=(1, 1)))
    assert n.data == dict(name="my_name", type="my_type", input_shape=(1, 1))


def test_attributes_are_merged_into_data_for_call_with_positional_args_only():
    n = Node.new("my_name", "my_type", dict(input_shape=(1, 1)))
    assert n.data == dict(name="my_name", type="my_type", input_shape=(1, 1))


def test_two_different_node_types_can_share_data() -> None:
    class NewNode(AbstractIRData):
        input_shape: MandatoryField[tuple[int, int]] = MandatoryField()

        @classmethod
        @overload
        def new(cls, input_shape: tuple[int, int]) -> "NewNode": ...

        @classmethod
        def new(cls, *args, **kwargs) -> "NewNode":
            return cls._do_new(*args, **kwargs)

    a = Node.new(name="my_name", type="my_type", attributes=dict(input_shape=(1, 3)))
    b = NewNode(a.data)
    b.input_shape = (4, 3)
    assert a.attributes["input_shape"] == b.input_shape


def test_nodes_inherit_mandatory_fields() -> None:
    class NewNode(Node):
        input_shape: MandatoryField[tuple[int, int]] = MandatoryField()

        @classmethod
        @overload
        def new(
            cls, name: str, type: str, input_shape: tuple[int, int]
        ) -> "NewNode": ...

        @classmethod
        def new(cls, *args, **kwargs) -> "NewNode":
            return cls._do_new(*args, **kwargs)

    n = NewNode.new(name="my_name", type="my_type", input_shape=(1, 3))
    assert n.input_shape == (1, 3)


def test_can_serialize_node_with_attributes():
    n = Node.new(name="my_node", type="my_type", attributes={"a": "b", "c": (1, 2)})
    assert dict(name="my_node", type="my_type", a="b", c=(1, 2)) == n.as_dict()


def test_can_deserialize_node_with_attributes():
    n = Node.new(name="my_node", type="my_type", attributes={"a": "b", "c": (1, 2)})
    assert n == Node.from_dict(dict(name="my_node", type="my_type", a="b", c=(1, 2)))


def test_import_path_does_not_matter_for_equality():
    n = Node.new(name="a", type="a_type", attributes=dict(a="b"))
    other = OtherNode.new(name="a", type="a_type", attributes=dict(a="b"))
    assert n == other


def test_set_transform_is_applied_when_calling_new() -> None:
    def set_transform(x: str) -> int:
        return int(x)

    def get_transform(x: int) -> str:
        return str(x)

    class NewNode(Node):
        input_shape: TransformableMandatoryField[int, str] = (
            TransformableMandatoryField(
                set_transform=set_transform,
                get_transform=get_transform,
            )
        )

        @classmethod
        @overload
        def new(cls, name: str, type: str, input_shape: str) -> "NewNode": ...

        @classmethod
        def new(cls, *args, **kwargs) -> "NewNode":
            return cls._do_new(*args, **kwargs)

    n = NewNode.new(name="my_name", type="my_type", input_shape="3")
    assert n.input_shape == "3"
