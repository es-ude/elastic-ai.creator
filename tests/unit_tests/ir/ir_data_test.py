from typing import Any

import pytest

from elasticai.creator.ir import (
    ReadOnlyField,
    RequiredField,
    read_only_field,
    static_required_field,
)
from elasticai.creator.ir.base.ir_data import IrData
from elasticai.creator.ir.base.ir_data_meta import IrDataMeta


def test_using_metaclass() -> None:
    class Node(metaclass=IrDataMeta, create_init=True):
        name: str

    n = Node(dict(name="x"))  # type: ignore
    assert "x" == n.name


def test_inheriting_with_metaclass() -> None:
    class Node(metaclass=IrDataMeta, create_init=True):
        name: str

    class MyNode(Node):
        type: str

    # __init__ not visible for type checker
    n = MyNode(dict(name="x", type="y"))  # type: ignore
    assert "y" == n.type


def test_inheriting_from_ir_base() -> None:
    class Node(IrData):
        name: str

    n = Node(dict(name="x"))
    assert "x" == n.name


def test_inherit_from_ir_base_child() -> None:
    class Node(IrData):
        name: str

    class Node2(Node):
        type: str

    n = Node2(data=dict(name="x", type="y"))
    assert "y" == n.type


def test_accidentally_writing_into_non_declared_field_raises_error() -> None:
    class Node(IrData):
        name: str

    n = Node(dict(name="x"))
    with pytest.raises(AttributeError):
        n.x = 5  # type: ignore


def test_inherited_class_contains_all_required_fields() -> None:
    class Node(IrData):
        name: str

    class Sub(Node):
        type: str

    assert dict(name=str, type=str).keys().mapping == Sub.required_fields


def test_get_missing_required_fields_returns_name_str() -> None:
    class Node(IrData):
        name: str

    class Sub(Node):
        type: str

    n = Sub(dict(type="a"))
    assert dict(name=str) == n.get_missing_required_fields()


def test_mandatory_fields_not_in_attributes() -> None:
    class Node(IrData):
        name: str
        type: str

    n = Node(dict(name="x", type="y", extra_field="f"))
    assert dict(extra_field="f") == n.attributes


def test_mandatory_fields_not_in_subclass_attributes() -> None:
    class Node(IrData):
        name: str

    class Sub(Node):
        type: str

    n = Sub(dict(name="a", type="b", extra="c"))
    assert dict(extra="c") == n.attributes


def test_defined_required_field_is_detected() -> None:
    class Node(IrData):
        width: RequiredField[str, int] = RequiredField(get_convert=int, set_convert=str)

    assert "width" in Node.required_fields


def test_defined_required_fields_type_is_detected() -> None:
    class Node(IrData):
        width: RequiredField[str, int] = RequiredField(get_convert=int, set_convert=str)

    assert issubclass(str, Node.required_fields["width"])


def test_can_increment_counter_saved_as_string() -> None:
    class Node(IrData):
        counter: RequiredField[str, int] = RequiredField(
            get_convert=int, set_convert=str
        )

    n = Node(dict(counter="1"))
    n.counter = n.counter + 2
    assert n.counter == 3
    assert n.data["counter"] == "3"


def test_read_only_field_is_detected() -> None:
    class Node(IrData):
        name: ReadOnlyField[str, str] = ReadOnlyField(lambda x: x)

    assert "name" in Node.required_fields


def test_attribute_is_read_only() -> None:
    class Node(IrData):
        name: ReadOnlyField[str, str] = ReadOnlyField(lambda x: x)

    n = Node(dict(name="x"))
    with pytest.raises(AttributeError):
        n.name = "y"


def test_right_oring_dict_with_attribute_does_not_add_hidden_fields() -> None:
    class Node(IrData):
        name: str
        type: str

    d: dict[str, Any] = dict()
    n = Node(dict(name="x", type="y", counter=1))
    d = d | n.attributes
    assert d == dict(counter=1)


def test_left_oring_dict_with_attribute_does_not_add_hidden_fields() -> None:
    class Node(IrData):
        name: str
        type: str

    d: dict[str, Any] = dict()
    n = Node(dict(name="x", type="y", counter=1))
    d = n.attributes | d
    assert d == dict(counter=1)


def test_inheriting_from_ir_data_without_annotated_fields_does_not_raise_exception():
    try:

        class Node(IrData):
            pass
    except KeyError:
        pytest.fail("did not expect an error")


class test_can_read_from_decorated_method_field:
    class Node(IrData):
        __slots__ = ("length_scaling",)

        def __init__(self, data: dict[str, Any]):
            super().__init__(data)
            self.length_scaling = 2

        @read_only_field
        def length(self, value: int) -> float:
            return value * self.length_scaling

    n = Node(dict(length=1))
    assert n.length == 2.0


class test_can_write_to_decorated_method_field:
    class Node(IrData):
        __slots__ = ("length_scaling",)

        def __init__(self, data: dict[str, Any]):
            super().__init__(data)

        @static_required_field
        @staticmethod
        def length(value: int) -> float:
            return value * 2

        @length.setter
        @staticmethod
        def _(value: float) -> int:
            return int(value / 2)

    n = Node(dict(length=1))
    n.length = 4.0
    assert n.data["length"] == 2
