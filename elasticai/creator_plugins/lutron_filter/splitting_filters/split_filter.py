from typing import Iterable, Iterator

from .._imports import FilterParameters


def find_channel_sizes_from_g1_to_g2(g1, g2, max_fan_in, w2) -> set[int]:
    sc_set = set()
    sc = g1
    while (sc // g2) * w2 <= max_fan_in:
        if sc % g2 == 0:
            sc_set.add(sc)
        sc += g1

    return sc_set


class Splittable:
    def __init__(self, fp: FilterParameters):
        self._fp = fp

    def __getattr__(self, item):
        return getattr(self._fp, item)

    def split_by_group(
        self, max_fan_in: int = 2, min_fan_in: int = 8
    ) -> set[tuple[FilterParameters, FilterParameters]]:
        return split_by_group(self, max_fan_in=max_fan_in, min_fan_in=min_fan_in)


def split_by_group(
    f: FilterParameters,
    min_fan_in: int = 2,
    max_fan_in: int = 8,
    max_fan_out: int = 8,
    min_fan_out: int = 8,
) -> set[tuple[FilterParameters, FilterParameters]]:
    if f.kernel_size > 12:
        raise ValueError("window size > 12 not supported")
    result = set()
    for fn in (split_by_group_end_with_pointwise, split_by_group_start_with_pointwise):
        result |= set(fn(f, max_fan_in=max_fan_in))
    max_fan_out_result = set()
    for split in result:
        if (
            min_fan_out <= split[0].out_channels <= max_fan_out
            and min(split[0].fan_in, split[1].fan_in) >= min_fan_in
        ):
            max_fan_out_result.add(split)

    return max_fan_out_result


def split_by_group_end_with_pointwise(
    original: FilterParameters, max_fan_in: int
) -> set[tuple[FilterParameters, FilterParameters]]:
    return _split_by_groups_with_kernel_sizes(
        original, max_fan_in, (original.kernel_size, 1)
    )


def split_by_group_start_with_pointwise(
    original: FilterParameters, max_fan_in: int
) -> set[tuple[FilterParameters, FilterParameters]]:
    return _split_by_groups_with_kernel_sizes(
        original, max_fan_in, (1, original.kernel_size)
    )


def _split_by_groups_with_kernel_sizes(
    original: FilterParameters, max_fan_in: int, kernel_sizes: tuple[int, int]
) -> set[tuple[FilterParameters, FilterParameters]]:
    fp = FilterParameters
    result: list[tuple[FilterParameters, FilterParameters]] = []

    w1 = kernel_sizes[0]
    w2 = kernel_sizes[1]

    def pair(g1, sc, g2):
        return (
            fp(
                kernel_size=w1,
                in_channels=original.in_channels,
                out_channels=sc,
                groups=g1,
            ),
            fp(
                kernel_size=w2,
                in_channels=sc,
                out_channels=original.out_channels,
                groups=g2,
            ),
        )

    def dividers(n: int) -> Iterator[int]:
        for i in range(1, n + 1):
            if n % i == 0:
                yield i

    def filter_fan_in(factors: Iterable[int], channel_size) -> Iterator[int]:
        for f in factors:
            if (channel_size // f) * w1 <= max_fan_in:
                yield f

    dg1 = set(filter_fan_in(dividers(original.in_channels), original.in_channels))
    dg2 = set(dividers(original.out_channels))

    def build_sc_set(g1: int, g2: int) -> set[int]:
        return find_channel_sizes_from_g1_to_g2(g1, g2, max_fan_in, w2=w2)

    for g1 in dg1:
        for g2 in dg2:
            for sc in build_sc_set(g1, g2):
                result.append(pair(g1, sc, g2))

    return set(result)


def filter_splits_with_min_fan_in(
    pairs: Iterable[tuple[FilterParameters, FilterParameters]], min_fan_in
):
    for a, b in pairs:
        if min(a.fan_in, b.fan_in) >= min_fan_in:
            yield a, b
