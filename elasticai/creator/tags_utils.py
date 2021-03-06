from typing import Any, Protocol, TypeVar, runtime_checkable

from elasticai.creator.mlframework import Module

T = TypeVar("T")


@runtime_checkable
class Tagged(Protocol):
    @property
    def elasticai_tags(self) -> dict:
        ...

    @elasticai_tags.setter
    def elasticai_tags(self, tags: dict) -> None:
        ...


T_co = TypeVar("T_co", bound=Module, covariant=True)


@runtime_checkable
class TaggedModule(Tagged, Module, Protocol):
    ...


def tag(module: TaggedModule, **new_tags: Any) -> TaggedModule:
    """Add tags to any object wrapping it in a TagWrapper if necessary

    new_tags will override possibly existing tags
    """
    old_tags = get_tags(module)
    tags = old_tags | new_tags
    module.elasticai_tags = tags
    return module


def get_tags(module: Any) -> dict:
    if isinstance(module, Tagged):
        return module.elasticai_tags
    return {}


def has_tag(module: Any, tag: str) -> bool:
    if isinstance(module, Tagged):
        return tag in module.elasticai_tags
    return False
