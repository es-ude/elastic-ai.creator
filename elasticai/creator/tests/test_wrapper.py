import functools
import inspect
import types
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable, TypeVar, Type, Callable
from unittest import TestCase

import wrapt
from wrapt import ObjectProxy, CallableObjectProxy

from elasticai.creator.tags_utils import Tagged

T = TypeVar("T")


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
    init = getattr(old_class, "__init__")

    @functools.wraps(init)
    def wrapped_init(*args, **kwargs):
        init(*args, **kwargs)
        setattr(args[0], attribute_name, attribute_value)

    setattr(old_class, "__init__", wrapped_init)
    wrapped = old_class

    return wrapped


def tag(**tags: Any) -> Callable[[type], Type[Tagged]]:
    # @wrapt.decorator
    # def wrapper(wrapped, instance, args, kwargs) -> Tagged:
    #     if instance is None and inspect.isclass(wrapped):
    #
    #         def elasticai_tags(self) -> dict[str, Any]:
    #             return self._elasticai_tags
    #
    #         wrapped.elasticai_tags = elasticai_tags
    #         obj = wrapped(*args, **kwargs)
    #         obj._elasticai_tags = tags
    #         return obj
    #     raise TypeError
    #
    class TagWrapper(CallableObjectProxy):
        def __init__(self, wrapped: type):
            super().__init__(wrapped)
            self._self_wrapper = wrapped
            self._elasticai_tags = tags

        @property
        def elasticai_tags(self) -> dict[str, Any]:
            return self._elasticai_tags

    return TagWrapper


@runtime_checkable
class P(Protocol):
    @abstractmethod
    def b(self):
        ...


class TestWrappingClasses(TestCase):
    def test_tag(self):
        @tag(test=1)
        class A:
            ...

        tagged = A()
        print(dir(A))
        self.assertEqual(1, tagged.elasticai_tags()["test"])
