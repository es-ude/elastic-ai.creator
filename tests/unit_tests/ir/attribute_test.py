import pytest

from elasticai.creator.ir.attribute import (
    AttributeMapping,
    attribute,
)


def test_nested_attribute_map_is_immutable() -> None:
    a = AttributeMapping(x=1, y=2)
    b = AttributeMapping(a=a, b=(3, 4))
    c = b.new_with(a=a.new_with(x=10))
    assert a["y"] == 2
    assert b["a"]["x"] == 1
    assert c["a"]["x"] == 10


def test_can_drop_key_from_mapping() -> None:
    a = AttributeMapping(x=1, y=2)
    a = a.drop("x")
    assert "x" not in a
    assert a["y"] == 2


def test_create_from_nested_dict() -> None:
    a = AttributeMapping.from_dict(dict(a="a", b=dict(c=[1, 2])))
    assert isinstance(a, AttributeMapping)
    assert isinstance(a["b"], AttributeMapping)
    assert a == AttributeMapping(a="a", b=AttributeMapping(c=(1, 2)))


def test_attribute_mapping_is_not_equal_to_dict():
    data = {"x": 1}
    assert data != AttributeMapping(x=1)


def test_replace_deeply_nested_value() -> None:
    c = AttributeMapping(a=AttributeMapping(b="c", c="c"))
    d = c.merge(AttributeMapping(a=AttributeMapping(b="d")))
    assert d["a"]["c"] == "c"
    assert d["a"]["b"] == "d"


def test_replace_nested_value_with_update_path() -> None:
    c = AttributeMapping(a=AttributeMapping(b="c", c="c"))
    d = c.update_path(("a", "b"), "d")
    assert d["a"]["c"] == "c"
    assert d["a"]["b"] == "d"


@pytest.mark.parametrize(
    "_input, expected",
    [
        (AttributeMapping(x=3, y=4), AttributeMapping(x=3, y=4)),
        (dict(x=1), AttributeMapping(x=1)),
        (dict(x=dict(y=1)), AttributeMapping(x=AttributeMapping(y=1))),
        (dict(x=[1, 2]), AttributeMapping(x=(1, 2))),
        (dict(x=[1, dict(y=2)]), AttributeMapping(x=(1, AttributeMapping(y=2)))),
        ([dict(x=1), dict(x=2)], (AttributeMapping(x=1), AttributeMapping(x=2))),
        ({1, 2, 3}, (1, 2, 3)),
        (range(3), (0, 1, 2)),
    ],
)
def test_can_convert_to_attributes(_input, expected) -> None:
    actual = attribute(_input)
    assert actual == expected


def test_can_combine_attribute_mapping_with_kwargs() -> None:
    expected = AttributeMapping(x=1, y=2)
    assert expected == attribute(dict(x=1), y=2)


def test_trying_to_call_attribute_with_int_and_kwargs_raises_type_error() -> None:
    with pytest.raises(TypeError):
        attribute(1, x=5)  # type: ignore


def test_can_convert_string() -> None:
    # need to check as strings are iterable
    assert "x" == attribute("x")


def test_can_call_attribute_with_kwargs_only_to_create_mapping() -> None:
    assert AttributeMapping(x=4) == attribute(x=4)
    assert AttributeMapping() == attribute()
    assert AttributeMapping() == attribute({})
