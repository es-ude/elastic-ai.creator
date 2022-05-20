import functools
from typing import Any, Callable

"""
The meta programming functions in this module are mostly used to implement the decorator software design patterns
that allow users to add support for elastic ai functionalities, such as *precomputing* to custom `torch.nn.Module`s.
"""


def add_method(method_name: str, method: Callable) -> Callable[[type], type]:
    """Python decorator for adding a method to a class definition.

    Returns:
        A class with a method called `method_name` added, that is implemented by `method`.

    """

    def wrapper(cls: type) -> type:
        return cls

    return wrapper


def add_instance_attribute(
    attribute_name: str, attribute: Any
) -> Callable[[type], type]:
    """Python decorator that adds an instance attribute to a class definition. This behaves as if the assignment was
    `self.attribute_name = attribute` was issued after the call to the original `__init__` function of the class.
    Returns:
        A new class being identical to the original class definition, except for the `__init__` function that executes above assignment
        as a last step

    """

    def adds(cls: type) -> type:
        init: Any = cls.__init__

        @functools.wraps(init)
        def wrapped_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            setattr(self, attribute_name, attribute)

        setattr(cls, "__init__", wrapped_init)
        return cls

    return adds
