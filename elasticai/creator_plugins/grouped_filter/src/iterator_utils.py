from typing import Iterable, Iterator, Sequence, TypeVar, overload

T = TypeVar("T")


@overload
def batched(vector: Sequence[T], size: int) -> Iterator[Sequence[T]]: ...


@overload
def batched(vector: Iterable[T], size: int) -> Iterator[Iterable[T]]: ...


def batched(vector, size):
    def make_batches(vector, size):
        steps = len(vector) // size
        for step in range(steps):
            yield vector[size * step : size * (step + 1)]

    def make_sequence(vector):
        is_iterable = hasattr(vector, "__iter__")
        is_sequence = hasattr(vector, "__len__") and hasattr(vector, "__getitem__")
        if not is_iterable and not is_sequence:
            raise ValueError("argument needs to be a Sequence or Iterable")
        if not is_sequence:
            return tuple(vector)
        return vector

    def handle_invalid_length(vector, size):
        if len(vector) % size != 0:
            raise ValueError

    vector = make_sequence(vector)
    handle_invalid_length(vector, size)
    yield from make_batches(vector, size)
