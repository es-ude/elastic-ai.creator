from typing import Generic, TypeVar, Union, runtime_checkable, Protocol, Any

T = TypeVar('T')


from torch.nn import Module


@runtime_checkable
class Unwrappable(Protocol[T]):
    def unwrap(self) -> T:
        raise NotImplementedError


@runtime_checkable
class HasElasticAITags(Protocol[T]):
    def elasticai_tags(self) -> dict:
        raise NotImplementedError


class TagWrapperMixin(Generic[T]):
    def __init__(self, wrapped: T, **tags: Any):
        super().__init__()
        self._wrapped = wrapped
        self._elasticai_tags = tags

    def unwrap(self):
        return self._wrapped

    def elasticai_tags(self) -> dict:
        return self._elasticai_tags


MaybeTagWrapper = Union[T, TagWrapperMixin[T]]


class ModuleTagWrapper(TagWrapperMixin[Module], Module):
    def __init__(self, module, **tags):
        super().__init__(wrapped=module, **tags)


def tag(module: MaybeTagWrapper, **new_tags: Any) -> ModuleTagWrapper:
    """Add tags to any object wrapping it in a TagWrapper if necessary

    new_tags will override possibly existing tags
    """
    old_tags = get_tags(module)
    module = unwrap(module)
    tags = old_tags | new_tags

    return ModuleTagWrapper(module=module, **tags)


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
