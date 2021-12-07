from typing import Generic, TypeVar, Union, runtime_checkable, Protocol, Any, Tuple, Sequence, TypeAlias

T = TypeVar('T')

@runtime_checkable
class Unwrappable(Protocol[T]):
    def unwrap(self) -> T:
        raise NotImplementedError


@runtime_checkable
class HasElasticAITags(Protocol[T]):
    def elasticai_tags(self) -> dict:
        raise NotImplementedError


class TagWrapper(Generic[T]):
    def __init__(self, wrapped: T, **tags: Any):
        super().__init__()
        self._wrapped = wrapped
        self._elasticai_tags = tags

    def __getattr__(self, item):
        return self._wrapped.__getattribute__(item)

    def unwrap(self):
        return self._wrapped

    def elasticai_tags(self) -> dict:
        return self._elasticai_tags


MaybeTagWrapper = Union[T, TagWrapper[T]]


def tag(module: MaybeTagWrapper, **new_tags: Any) -> TagWrapper[T]:
    """Add tags to any object wrapping it in a TagWrapper if necessary

    new_tags will override possibly existing tags
    """
    old_tags = get_tags(module)
    module = unwrap(module)
    tags = old_tags | new_tags
    return TagWrapper(wrapped=module, **tags)


def unwrap(module: Union[T, Unwrappable[T]]) -> T:
    if isinstance(module, Unwrappable):
        return module.unwrap()
    return module


def get_tags(module: MaybeTagWrapper) -> dict:
    if isinstance(module, HasElasticAITags):
        return module.elasticai_tags()
    return {}


def has_tag(module: MaybeTagWrapper, tag_name) -> bool:
    return tag_name in get_tags(module)
