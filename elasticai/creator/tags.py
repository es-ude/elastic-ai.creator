from typing import Generic, TypeVar, Union, runtime_checkable, Protocol, Any, Tuple, Sequence

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


def tag_precomputed(module: Union[T, TagWrapper[T]],
                    input_shape: Tuple[int],
                    input_domain: Sequence[float]) -> TagWrapper[T]:
    return tag(module, precomputed={'input_shape': input_shape, 'input_domain': input_domain})


def tag(module: Union[T, TagWrapper[T]], **new_tags: Any) -> TagWrapper[T]:
    old_tags = get_tags(module)
    module = unwrap(module)
    tags = old_tags | new_tags
    return TagWrapper(wrapped=module, **tags)


def unwrap(module: Union[T, Unwrappable[T]]) -> T:
    if isinstance(module, Unwrappable):
        return module.unwrap()
    return module


def get_tags(module: Union[T, TagWrapper[T]]) -> dict:
    if isinstance(module, HasElasticAITags):
        return module.elasticai_tags()
    return {}
