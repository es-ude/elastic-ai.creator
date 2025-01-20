import operator
from functools import reduce
from typing import TypeAlias, TypeGuard, overload

ShapeTuple: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]


def is_shape_tuple(values) -> TypeGuard[ShapeTuple]:
    max_num_values = 3
    return len(values) <= max_num_values


class Shape:
    @overload
    def __init__(self, width: int, /) -> None: ...

    @overload
    def __init__(self, depth: int, width: int, /) -> None: ...

    @overload
    def __init__(self, depth: int, width: int, height: int, /) -> None: ...

    def __init__(self, *values: int) -> None:
        """values are interpreted as one of the following:
        - width
        - depth, width
        - depth, width, height

        Usually width is kernel_size, depth is channels
        """

        if is_shape_tuple(values):
            self._data = values
        else:
            raise TypeError(f"taking at most three ints, given {values}")

    @classmethod
    def from_tuple(cls, values: ShapeTuple) -> "Shape":
        return cls(*values)  # type ignore

    def to_tuple(self) -> ShapeTuple:
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    def size(self) -> int:
        return reduce(operator.mul, self._data, 1)

    def ndim(self) -> int:
        return len(self._data)

    @property
    def depth(self) -> int:
        return self._data[0]

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self._data == other
        if isinstance(other, Shape):
            return self._data == other._data

        return False

    @property
    def width(self) -> int:
        if len(self._data) > 1:
            return self._data[1]
        else:
            return 1

    @property
    def height(self) -> int:
        if len(self._data) > 2:
            return self._data[2]
        return 1

    def __repr__(self) -> str:
        match self._data:
            case (width,):
                return f"Shape({width=})"
            case (depth, width):
                return f"Shape({depth=}, {width=})"
            case (depth, width, height):
                return f"Shape({depth=}, {width=}, {height=})"
            case _:
                return f"Shape({self._data})"


class FilterParameters:
    """
    Represents all necessary parameters to implement a (grouped) 1d filter
    without padding.
    args:
        input_size: Denotes the spatial input size of the filter, consequently we assume `kernel_size <= input_size`.
            However, in some cases `input_size` might not be known, until this changes we set it to zero.
            Typically, we need to run an inference once to set the input_size fields.
        kernel_size: Either the size of the kernel in case of e.g. conv1d or pooling layers, or `-1` in case of a layer
            that expects a flat tensor (e.g., linear layer).
            IMPORTANT: The way we represent linear layers is expected to change in the future.
        input_channels: The number of input channels in case of a 1d filter. In case of a flat input tensor, e.g., linear layer
    """

    _field_names = (
        "out_channels",
        "in_channels",
        "kernel_size",
        "groups",
        "stride",
        "input_size",
        "output_size",
    )

    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        stride: int = 1,
        input_size: int | None = None,
        output_size: int = 1,
    ):
        self.kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._groups = groups
        self._check_group_validity()
        if input_size is None:
            input_size = kernel_size
        self.input_size = input_size
        self.output_size = output_size

    def _check_group_validity(self):
        if not (
            self.in_channels % self.groups == 0 and self.out_channels % self.groups == 0
        ):
            raise ValueError(
                f"groups has to be a divider of in_channels and out_channels, but found in_channels: {self.in_channels}, out_channels: {self.out_channels}, groups: {self.groups}"
            )

    @property
    def fan_in(self) -> int:
        return self.in_channels // self.groups * self.kernel_size

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def num_steps(self) -> int:
        return (self.input_size - self.kernel_size) // self.stride + 1

    @property
    def groups(self) -> int:
        return self._groups

    @groups.setter
    def groups(self, groups: int):
        self._groups = groups
        self._check_group_validity()

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @in_channels.setter
    def in_channels(self, in_channels: int):
        self._in_channels = in_channels
        self._check_group_validity()

    @property
    def in_channels_per_group(self) -> int:
        return self.in_channels // self.groups

    @property
    def out_channels_per_group(self) -> int:
        return self.out_channels // self.groups

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @out_channels.setter
    def out_channels(self, out_channels: int):
        self._out_channels = out_channels
        self._check_group_validity()

    @property
    def total_skipped_inputs_per_step(self) -> int:
        return self.stride * self.in_channels

    def get_in_channels_by_group(self) -> tuple[tuple[int, ...], ...]:
        return self._get_channels_by_group(self.in_channels_per_group)

    def get_out_channels_by_group(self) -> tuple[tuple[int, ...], ...]:
        return self._get_channels_by_group(self.out_channels_per_group)

    def _get_channels_by_group(self, channels_per_group):
        def get_channels_for_group(g):
            return tuple(range(g * channels_per_group, (g + 1) * channels_per_group))

        return tuple((get_channels_for_group(g) for g in range(self.groups)))

    def with_groups(self, groups) -> "FilterParameters":
        return self.__class__(
            kernel_size=self.kernel_size,
            groups=groups,
            out_channels=self.out_channels,
            in_channels=self.in_channels,
        )

    def _value_dict(self):
        return {f: getattr(self, f) for f in self._field_names}

    def as_dict(self):
        return self._value_dict()

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: d[k] for k in cls._field_names})

    def __repr__(self) -> str:
        params = ", ".join([f"{k}={v}" for k, v in self._value_dict().items()])
        return f"FilterParameters({params})"

    def __hash__(self):
        return hash(tuple(self._value_dict().values()))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return hash(self) == hash(other)
