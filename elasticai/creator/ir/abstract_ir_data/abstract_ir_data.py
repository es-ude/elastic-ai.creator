import inspect
import sys
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

from elasticai.creator.ir.attribute import AttributeT

from ._attributes_descriptor import _AttributesDescriptor
from .mandatory_field import MandatoryField, TransformableMandatoryField

if sys.version_info.minor > 10:
    from typing import Self
else:
    Self = TypeVar("Self", bound="AbstractIrData")


class AbstractIRData:
    """
    This class should provide a way to easily create new wrappers around dictionaries.

    It is supposed to be used together with the `MandatoryField` class.
    Every child of `AbstractIRData` is expected to have a constructor that takes a dictionary.
    That dictionary is not copied, but instead can be shared with other Node classes.

    Most of the private functions in this class deal with handling arguments of the classmethod
    `new`, that is used to create new nodes from scratch.

    The `attributes` attribute of the class provides a dict-like object, that hides all keys that are associated
    with mandatory fields.

    The purpose of this class is to provide a way to easily write new wrappers around dictionaries, that let us customize
    access, while still allowing static type annotations.
    """

    __slots__ = ("data",)
    attributes = _AttributesDescriptor(set())

    def __init__(self: Self, data: dict[str, AttributeT]):
        """IMPORTANT: Do not override this. If you want to create a function that creates new nodes of your subtype,
        override the `new` method instead.
        """
        for k in self._mandatory_fields():
            if k not in data:
                raise ValueError(f"Missing mandatory field {k}")
        self.data = data

    @classmethod
    def _do_new(cls, *args, **kwargs) -> Self:
        """This is here for your convenience to be called in `new`."""
        cls.__validate_arguments(args, kwargs)
        data = cls.__turn_arguments_into_data_dict(args, kwargs)
        return cls(data)

    @classmethod
    @abstractmethod
    def new(cls, *args, **kwargs) -> Self:
        """Create a new node by creating a new dictionary from args and kwargs.

        Use this for creation of new nodes from inline code. This is typically also where you want to provide
        type hints for users via `@overload`. You can delegate to the `_do_new()` method of `BaseNode`
        """
        ...

    def as_dict(self: Self) -> dict[str, AttributeT]:
        return self.data

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, AttributeT]) -> Self:
        return cls(data)

    def __eq__(self: Self, other: object) -> bool:
        if hasattr(other, "data") and isinstance(other.data, dict):
            return self.data == other.data
        else:
            return False

    @classmethod
    def __turn_arguments_into_data_dict(
        cls, args: tuple[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        data = cls.__extract_attributes_from_args(args, kwargs)
        data.update(cls.__get_kwargs_without_attributes(kwargs))
        data.update(cls.__get_args_as_kwargs(args))
        cls.__transform_args_with_mandatory_fields(data)
        return data

    @classmethod
    def __get_mandatory_field_descriptors(
        cls,
    ) -> Iterable[tuple[str, TransformableMandatoryField]]:
        for c in reversed(inspect.getmro(cls)):
            for a in c.__dict__:
                if (
                    not a.startswith("__")
                    and not a.endswith("__")
                    and isinstance(c.__dict__[a], TransformableMandatoryField)
                ):
                    yield a, c.__dict__[a]

    @classmethod
    def _mandatory_fields(cls) -> tuple[str, ...]:
        return tuple(name for name, _ in cls.__get_mandatory_field_descriptors())

    def __attribute_keys(self: Self):
        return tuple(k for k in self.data.keys() if k not in self._mandatory_fields())

    def __repr__(self: Self):
        mandatory_fields_repr = ", ".join(
            f"{k}={self.data[k]}" for k in self._mandatory_fields()
        )
        attributes = ", ".join(
            f"'{k}': '{self.data[k]}'" for k in self.__attribute_keys()
        )
        return (
            f"{self.__class__.__name__}({mandatory_fields_repr},"
            f" attributes={attributes})"
        )

    @classmethod
    def __validate_arguments(cls, args: tuple[Any], kwargs: dict[str, Any]):
        num_total_args = len(args) + len(kwargs)
        if num_total_args not in (
            len(cls._mandatory_fields()),
            len(cls._mandatory_fields()) + 1,
        ):
            raise ValueError(
                f"allowed arguments are {cls._mandatory_fields()} and attributes, but"
                f" passed args: {args} and kwargs: {kwargs}"
            )
        argument_names_in_args = set(k for k, _ in zip(cls._mandatory_fields(), args))
        arguments_specified_twice = argument_names_in_args.intersection(kwargs.keys())
        if len(arguments_specified_twice) > 0:
            raise ValueError(f"arguments specified twice {arguments_specified_twice}")

    @classmethod
    def __extract_attributes_from_args(
        cls, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        data = dict()
        if "attributes" in kwargs:
            if callable(kwargs["attributes"]):
                data.update(kwargs["attributes"]())
            else:
                data.update(kwargs["attributes"])
        elif len(args) + len(kwargs) == len(cls._mandatory_fields()) + 1:
            attributes = args[-1]
            data.update(attributes)
        return data

    @classmethod
    def __get_kwargs_without_attributes(
        cls, kwargs: dict[str, Any]
    ) -> dict[str, AttributeT]:
        kwargs = {k: v for k, v in kwargs.items() if k != "attributes"}
        return kwargs

    @classmethod
    def __transform_args_with_mandatory_fields(cls, args: dict[str, Any]) -> None:
        set_transforms = cls.__get_field_transforms()
        for k in args:
            if k in set_transforms:
                args[k] = set_transforms[k](args[k])

    @classmethod
    def __get_field_transforms(cls) -> dict[str, Callable[[Any], AttributeT]]:
        return {k: v.set_transform for k, v in cls.__get_mandatory_field_descriptors()}

    @classmethod
    def __get_args_as_kwargs(cls, args: tuple[Any]) -> Iterable[tuple[str, Any]]:
        return zip(cls._mandatory_fields(), args)
