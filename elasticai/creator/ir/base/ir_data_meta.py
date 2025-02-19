from collections.abc import MutableMapping
from types import GenericAlias, MappingProxyType, resolve_bases
from typing import Any, cast

from .required_field import SimpleRequiredField, is_required_field


class IrDataMeta(type):
    """
    This implementation tries to find a good compromise between
    avoiding boiler plate code and compatibility with
    static type checkers.

    The metaclass provides automatic generation of an init function
    and will convert any type annotations with `Attribute` types into corresponding
    `MandatoryField`s.
    To **not** generate an init function define your class like this

    ```python
    class C(metaclass=IrDataMeta, create_init=False):
      name: str
    ```

    We recommend inheriting from the `IrData` class instead as this provides
    useful checks for data integrity and an init function that will be detected
    by static type checkers.

    IMPORTANT: Type annotations for names starting with `_` are excluded from this!


    NOTE:
    Other options of implementation have been considered and tried:

    - dynamically add fields and init function after class creation
      - (-)(-) neither fields nor data attribute detected by static type checkers
      - (-) __set_name__ callbacks are not triggered
      - (+) easier to understand
      - (-) __slots__ need to be defined manually by the user, which will certainly lead
        to hard to find bugs.
    - a decorator that builds a new class taking the decorated class as a prototype
      - (+)(-) type hints for fields are picked up
      - (-) a lot of boiler plate to make type checkers detect init and other attributes
      - (+) if it wasn't for the type checks, this would be easy to understand and maintain
        and have the least amount of boiler plate
    """

    @property
    def required_fields(self) -> MappingProxyType[str, type]:
        """
        In `IrData` and its children this will be available
        as a class property. We add it here because python3.13 removed
        the ability to combine classmethod and property
        """
        return self._fields.keys().mapping  # type: ignore

    @classmethod
    def __prepare__(cls, name, bases, create_init=False, **kwds):
        """
        called before parsing class body and thus before metaclass instantiation,
        hence this is a classmethod
        """
        namespace = super().__prepare__(name, bases, **kwds)
        cls.__add_data_slot(namespace)
        if create_init:
            cls.__add_init(namespace)
        namespace["_fields"] = {}
        return namespace

    @classmethod
    def __add_init(cls, namespace):
        def init(self, data):
            self.data = data

        namespace["__init__"] = init

    def __new__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds
    ) -> "IrDataMeta":
        """
        Create and return a new IrDataMeta object.
        This is the type for new IrData classes.
        Fields defined in the namespace are fields of the class, not fields
        of the instantiated object. E.g., the deriving class will feature
        an `__init__` method, the owner of this method is the class, i.e., an object
        of type `IRDataMeta`. When accessing such a method attribute via the `dot`
        from an object with metaclass `IrDataMeta`, the method will be bound
        to that specific instance on the fly.

        As such, when dynamically creating the mandatory fields below,
        this happens once per class definition, not per instantiation
        of such a class.

        """
        inherited_fields = cls.__get_inherited_fields(bases)
        generated_fields = cls.__get_fields_to_be_generated(namespace)
        user_defined_fields = cls.__get_user_defined_fields_from_annotations(namespace)
        for name in generated_fields:
            namespace[name] = SimpleRequiredField()
        namespace["_fields"] = inherited_fields | generated_fields | user_defined_fields
        return super().__new__(cls, name, bases, namespace)

    @staticmethod
    def __get_inherited_fields(bases):
        inherited_fields = {}
        for o in resolve_bases(bases):
            if isinstance(o, IrDataMeta):
                inherited_fields.update(o._fields)  # type: ignore
        return inherited_fields

    @staticmethod
    def __add_data_slot(namespace: MutableMapping[str, object]) -> None:
        namespace["__slots__"] = ("data",) + cast(
            tuple, namespace.get("__slots__", tuple())
        )

    @staticmethod
    def __get_user_defined_fields_from_annotations(
        namespace: dict[str, Any],
    ) -> dict[str, type]:
        annotations = namespace.get("__annotations__", {})
        fields = dict()
        for name, annotation in annotations.items():
            if not name.startswith("_") and name in namespace:
                item = namespace[name]
                if is_required_field(item):
                    stored_type = annotation.__args__[0]
                    fields[name] = stored_type
        return cast(dict[str, type], fields)

    @staticmethod
    def __get_fields_to_be_generated(
        namespace: dict[str, Any],
    ) -> dict[str, type]:
        annotations = namespace.get("__annotations__", {})
        fields: dict[str, type] = dict()
        for name, annotation in annotations.items():
            if name.startswith("_") or name in namespace:
                continue
            fields[name] = IrDataMeta.__get_annotations_type(annotation)
        return fields

    @staticmethod
    def __get_annotations_type(annotation) -> type:
        if isinstance(annotation, str):
            annotation = eval(annotation, globals(), locals())
        if isinstance(annotation, GenericAlias):
            _type = annotation.__origin__
        else:
            _type = annotation
        return _type
