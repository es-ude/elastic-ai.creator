import functools
import types
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable, TypeVar, Type
from unittest import TestCase

T = TypeVar("T")


def add_attribute_to_class(
    old_class: Type[T], new_class_name: str, name: str, value: Any
) -> Type[T]:
    def body(namespace):
        for attr in old_class.__dict__:
            namespace[attr] = getattr(old_class, attr)

    class_copy = types.new_class(
        new_class_name, bases=old_class.__bases__, exec_body=body
    )
    setattr(class_copy, name, value)
    return class_copy


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


def tag(**tags: Any):

    decorator = functools.partial(
        add_instance_attribute_to_class_def,
        attribute_name="_elasticai_tags",
        attribute_value=tags,
    )
    return decorator


@runtime_checkable
class P(Protocol):
    @abstractmethod
    def b(self):
        ...


class A:
    ...


class B(A):
    ...


class TestWrappingClasses(TestCase):
    def test_base(self):
        class A:
            ...

        MyClass = add_attribute_to_class(A, "MyClass", "b", 1)
        self.assertEqual(MyClass.b, 1)

    def test_class_method_protocol(self):
        class A:
            pass

        @runtime_checkable
        class B(Protocol):
            @abstractmethod
            def b(self):
                ...

        def b(s):
            pass

        C = add_attribute_to_class(A, "C", "b", b)
        a = C()

        def test_static_type_checks(obj: B):
            pass

        test_static_type_checks(a)
        self.assertTrue(isinstance(a, B))
        self.assertTrue(not isinstance(A(), B))
