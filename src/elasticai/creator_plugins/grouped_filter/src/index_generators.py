from itertools import chain
from typing import Iterator

from .filter_params import FilterParameters
from .iterator_utils import batched
from .tensor import reshape_flat_CxN_groupwise


def channelwise_interleaved(length, channels):
    size_per_channel = length // channels
    indices_per_channel = batched(tuple(range(length)), size_per_channel)
    yield from chain.from_iterable(zip(*indices_per_channel))


def groupwise(length, channels, groups):
    yield from reshape_flat_CxN_groupwise(tuple(range(length)), channels, groups)


def generate_deinterleaved_indices(length, channels, channel_size):
    part_size = length // (channels * channel_size)
    by_channel = tuple(batched(tuple(range(length)), channel_size))
    by_part = tuple(batched(by_channel, part_size))
    num_parts = length // part_size // channel_size
    for c in range(part_size):
        for s in range(num_parts):
            yield from by_part[s][c]


def sliding_window(steps, size, stride):
    start = 0
    while steps > 0:
        yield range(start, start + size)
        start = start + stride
        steps = steps - 1


class GroupedFilterIndexGenerator:
    """Generates a sequence of indices as needed to apply a grouped filter kernel to a 1d signal.
    That means for a one dimensional input tensor with interleaved channels of length `spatial_size*in_channels` the generator
    can produce the correct indices for each group or step depending on the given filter parameters
    kernel size, number of input channels and stride.
    E.g.,
    >>> p = FilterParameters(in_channels=1, out_channels=1, kernel_size=2, stride=1)
    >>> g = GroupedFilterIndexGenerator(3, p)
    >>> for g_idx, group in enumerate(g.groups()):
    >>>     print("group id: ", g_idx)
    >>>     for s_idx, step in enumerate(group.steps()):
    >>>         print("step ", s_idx, ': ', list(step))
    >>>
    ... "group id: 0"
    ... "step 0: [0, 1]"
    ... "step 1: [1, 2]"
    """

    def __init__(self, params: FilterParameters):
        self._params = params

    def groups(self) -> Iterator["Group"]:
        yield from (Group(self._params, x) for x in range(self._params.groups))

    def steps(self) -> Iterator["Step"]:
        yield from (Step(self._params, x) for x in range(self._params.num_steps))

    def as_tuple_by_steps(self):
        return tuple(tuple(tuple(g) for g in step.groups()) for step in self.steps())

    def as_tuple_by_groups(self):
        return tuple(tuple(tuple(s) for s in group.steps()) for group in self.groups())


class Step:
    def __init__(self, params: FilterParameters, step_id: int):
        self._p = params
        self._step_id = step_id

    def groups(self) -> Iterator["IndicesForOperation"]:
        for group in range(self._p.groups):
            yield IndicesForOperation(
                self._p,
                self._step_id,
                group,
            )


class Group:
    def __init__(self, params: FilterParameters, group_id: int):
        self._p = params
        self._group_id = group_id

    def steps(self) -> Iterator["IndicesForOperation"]:
        for step in range(self._p.num_steps):
            yield IndicesForOperation(self._p, step, self._group_id)


class IndicesForOperation:
    def __init__(self, params: FilterParameters, step_id, group_id):
        self._p = params
        self._step = step_id
        self._group = group_id

    def __iter__(self):
        step_offset = self._p.total_skipped_inputs_per_step * self._step
        group_offset = self._group * self._p.in_channels_per_group
        offset = step_offset + group_offset
        for tap in range(self._p.kernel_size):
            for c in range(self._p.in_channels_per_group):
                yield offset + c + tap * self._p.in_channels


def unrolled_grouped_convolution(
    spatial_input_size: int,
    window_size: int,
    in_channels: int,
    groups: int,
    stride: int,
):
    """
    Generates indices for a grouped convolutional filter.
    The structure is (Steps, Groups, window_size*channels_per_group)
    """
    steps = (spatial_input_size - window_size) // stride + 1
    channels_per_group = in_channels // groups
    skipped_per_stride = stride * in_channels

    def generate_group(stride_offset, group):
        group_offset = stride_offset + group * channels_per_group

        for tap in range(window_size):
            for c in range(channels_per_group):
                yield group_offset + c + tap * in_channels

    def generate_step(step):
        for group in range(groups):
            yield generate_group(stride_offset=step * skipped_per_stride, group=group)

    for s in range(steps):
        yield generate_step(s)
