from elasticai.creator.ir.attribute import AttributeMapping


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
