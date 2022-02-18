from typing import Any, TypeVar, Protocol, Iterable, runtime_checkable

from torch import Tensor

T = TypeVar("T")


@runtime_checkable
class Tagged(Protocol):
    def elasticai_tags(self) -> dict:
        pass


@runtime_checkable
class ModuleProto(Protocol):
    @property
    def training(self) -> bool:
        pass

    def extra_repr(self) -> str:
        pass

    def named_children(self) -> Iterable[tuple[str, "ModuleProto"]]:
        pass

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        pass


@runtime_checkable
class TaggedModule(ModuleProto, Tagged, Protocol):
    pass


def tag(module: ModuleProto, **new_tags: Any) -> TaggedModule:
    """Add tags to any object wrapping it in a TagWrapper if necessary

    new_tags will override possibly existing tags
    """
    old_tags = get_tags(module)
    tags = old_tags | new_tags
    module._elasticai_tags = tags

    def get_tags_method(self=module) -> dict:
        return self._elasticai_tags

    module.elasticai_tags = get_tags_method
    return module


def get_tags(module: Any) -> dict:
    if isinstance(module, Tagged):
        return module.elasticai_tags()
    return {}


def has_tag(module: Any, tag: str) -> bool:
    if isinstance(module, Tagged):
        return tag in module.elasticai_tags()
    return False
