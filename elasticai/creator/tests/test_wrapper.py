import functools
import types
from typing import Union, Any
from types import FunctionType
from unittest import TestCase
from wrapt import ObjectProxy


def add_instance_attribute_to_class_def(
    old_class: type, attribute_name: str, attribute_value: Any
):
    """
    Supposed to be used to construct decorators for extending class definitions.
    Essentially the function will use the namespace of `old_class` to create a new class object.
    Args:
        old_class:
        attribute_name:
        attribute_value:

    Returns:

    """

    @functools.wraps(old_class.__init__)
    def wrapped_init(*args, **kwargs):
        old_class.__init__(*args, **kwargs)
        setattr(args[0], attribute_name, attribute_value)

    def class_body(namespace):
        for key in old_class.__dict__:
            namespace[key] = old_class.__dict__[key]
        namespace["__init__"] = wrapped_init

    wrapped = types.new_class(
        "Wrapper", bases=old_class.__bases__, exec_body=class_body
    )

    return wrapped


def add_method_to_class(to_be_wrapped: type, function: FunctionType):
    return functools.partial(
        add_instance_attribute_to_class_def, function.__name__, function
    )


def tag(**tags: Any):

    decorator = functools.partial(
        add_instance_attribute_to_class_def,
        attribute_name="_elasticai_tags",
        attribute_value=tags,
    )
    return decorator


class Base:
    pass


@tag(test=1)
class Wrapped(Base):
    def __init__(self):
        self.a = 1

    def b(self):
        return 2


class TestWrappingClasses(TestCase):
    def test_base(self):

        w = Wrapped()
        self.assertEqual({"test": 1}, w._elasticai_tags)
        self.assertTrue(not hasattr(Wrapped, "_elasticai_tags"))
        self.assertTrue(isinstance(w, Base))
