from typing import Any, TypeVar, Generic

T = TypeVar("T")


class HasTag(Generic[T]):
    def elasticai_tags(self) -> dict:
        pass


def tag(module: T, **new_tags: Any) -> HasTag:
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
    if hasattr(module, "elasticai_tags"):
        return module.elasticai_tags()
    return {}


def has_tag(module: Any, tag_name) -> bool:
    return tag_name in get_tags(module)
