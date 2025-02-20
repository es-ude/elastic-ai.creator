from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar

_T = TypeVar("_T")


class Shape(Protocol):
    width: int

    def size(self) -> int: ...


class Node(Protocol):
    name: str
    implementation: str
    input_shape: Shape
    output_shape: Shape


def _for_each_produce_multiple_lines(
    fn: Callable[[_T], Iterable[str]],
) -> Callable[[Iterable[_T]], list[str]]:
    def _wrapped(items: Iterable[_T]) -> list[str]:
        result: list[str] = list()
        for item in items:
            result.extend(fn(item))

        return result

    return _wrapped


def _for_each_produce_single_line(
    fn: Callable[[_T], str],
) -> Callable[[Iterable[_T]], list[str]]:
    @_for_each_produce_multiple_lines
    def _wrapped(item: _T) -> Iterable[str]:
        yield fn(item)

    return _wrapped


@_for_each_produce_multiple_lines
def connect_data_signals(
    edge: tuple[str, str, Sequence[tuple[int, int]] | tuple[str, str]],
):
    def get_args(r) -> tuple[int] | tuple[int, int]:
        args = tuple(int(n) for n in r.strip("range(").rstrip(")").split(","))
        if len(args) not in (1, 2):
            raise ValueError(f"Invalid range: {r}")
        return args  # pyright: ignore

    source, dst, wiring = edge

    if len(wiring) >= 1 and isinstance(wiring[0], tuple) and len(wiring[0]) == 2:
        for i, j in wiring:
            yield f"d_in_{dst}({j}) <= d_out_{source}({i});"
        return
    if len(wiring) == 0:
        yield f"d_in_{dst} <= d_out_{source};"
        return
    elif (
        len(wiring) == 2
        and isinstance(wiring[0], str)
        and isinstance(wiring[1], str)
        and wiring[0].startswith("range")
        and wiring[1].startswith("range")
    ):
        src_args = get_args(wiring[0])
        dst_args = get_args(wiring[1])
        if len(src_args) <= 2 and len(dst_args) <= 2:
            if len(src_args) == 1:
                src_args = (0, src_args[0])
            if len(dst_args) == 1:
                dst_args = (0, dst_args[0])
            yield f"d_in_{dst}({dst_args[1] - 1} downto {dst_args[0]}) <= d_out_{source}({src_args[1] - 1} downto {src_args[0]});"
            return
        else:
            wiring = zip(range(*src_args), range(*dst_args))
    for i, j in wiring:  # pyright: ignore
        yield f"d_in_{dst}({j}) <= d_out_{source}({i});"


def _define_signal_vector(name, length):
    return f"signal {name}: std_logic_vector({length}-1 downto 0) := (others => '0');"


@_for_each_produce_single_line
def define_input_data_signals(instance):
    return _define_signal_vector(f"d_in_{instance.name}", instance.input_shape.size())


@_for_each_produce_single_line
def define_output_data_signals(instance: Node):
    return _define_signal_vector(f"d_out_{instance.name}", instance.output_shape.size())


@_for_each_produce_single_line
def instantiate_bufferless(instance: Node):
    return f"{instance.name} : entity work.{instance.implementation}(rtl) port map (x => x_{instance.name}, y => y_{instance.name}, enable => enable);"
