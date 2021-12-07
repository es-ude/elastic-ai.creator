from elasticai.creator.tags_utils import tag, TagWrapper
from typing import TypeVar, Tuple, Sequence, Union

T = TypeVar('T')


def precomputed(module: Union[T, TagWrapper[T]],
                input_shape: Tuple[int],
                input_domain: Sequence[float]) -> TagWrapper[T]:
    return tag(module, precomputed={'input_shape': input_shape, 'input_domain': input_domain})
