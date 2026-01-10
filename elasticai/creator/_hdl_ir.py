import operator
from collections.abc import Callable, Iterable, Sequence
from functools import reduce
from typing import Any, Protocol, TypeGuard, cast, overload

import elasticai.creator.ir.ir_v2 as ir

type ShapeTuple = tuple[int] | tuple[int, int] | tuple[int, int, int]


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
    def from_tuple(cls, values: ShapeTuple | list[int]) -> "Shape":
        return cls(*values)

    def to_tuple(self) -> ShapeTuple:
        return self._data

    def to_list(self) -> list[int]:
        return list(self.to_tuple())

    def __getitem__(self, item):
        return self._data[item]

    def size(self) -> int:
        return reduce(operator.mul, self._data, 1)

    def ndim(self) -> int:
        return len(self._data)

    @property
    def depth(self) -> int:
        if len(self._data) > 1:
            return self._data[0]
        return 1

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self._data == other
        if isinstance(other, Shape):
            return self._data == other._data

        return False

    @property
    def width(self) -> int:
        if len(self._data) > 1:
            return self._data[1]  # ty: ignore
        else:
            return self._data[0]

    @property
    def height(self) -> int:
        if len(self._data) > 2:
            return self._data[2]  # ty: ignore
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


class Node(ir.Node, Protocol):
    """Extending ir.core.Node to an hdl specific node.

    This node contains all knowledge that we need to create
    and use an instance of an hdl component. However, this becomes
    a little bit complicated because vhdl differentiates between

    Attributes:

    implementation:: The name of the implementation, e.g., entity name in vhdl or module name for verilog, will be used to derive the architecture name.
        E.g., if the implementation is `"adder"`, we will instantiate the entity `work.adder(rtl)`.
        CAUTION: This behaviour is subject to change. Future versions might require the full entity name
    """

    @property
    def implementation(self) -> str: ...

    @property
    def input_shape(self) -> Shape: ...

    @property
    def output_shape(self) -> Shape: ...


def _type_check[T](item: Any, t: type[T]) -> T:
    if isinstance(item, t):
        return item
    else:
        raise TypeError(f"Expected type {t} but found {type(item)}")


class NodeImpl(ir.NodeImpl):
    @staticmethod
    def _shape_check(item: Any) -> tuple[int, ...]:
        if is_shape_tuple(item):
            return item
        else:
            raise TypeError(
                f"expected input_shape to be of type tuple[int, ...] but found {type(item)}"
            )

    @property
    def implementation(self) -> str:
        return _type_check(self.attributes.get("implementation", "<none>"), str)

    @property
    def input_shape(self) -> Shape:
        shape = self.attributes.get("input_shape", tuple())
        return Shape(*self._shape_check(shape))

    @property
    def output_shape(self) -> Shape:
        shape = self.attributes.get("output_shape", tuple())
        return Shape(*self._shape_check(shape))


class Edge(ir.Edge, Protocol):
    @property
    def src_dst_indices(self) -> tuple[tuple[int | str, int | str], ...]: ...


class EdgeImpl(ir.EdgeImpl):
    @property
    def src_dst_indices(self) -> tuple[tuple[int | str, int | str], ...]:
        indices = self.attributes.get("src_dst_indices", tuple())
        return _type_check(indices, tuple)


class DataGraph(ir.DataGraph[Node, Edge], Protocol):
    @property
    def name(self) -> str: ...

    @property
    def type(self) -> str: ...


class DataGraphImpl(ir.DataGraphImpl[Node, Edge]):
    @property
    def name(self) -> str:
        return _type_check(self.attributes.get("name", None), str)

    @property
    def type(self) -> str:
        return _type_check(self.attributes.get("type", "<undefined>"), str)


class IrFactory:
    def node(
        self,
        name: str,
        attributes: ir.AttributeMapping = ir.AttributeMapping(),
        /,
        type: str | None = None,
        input_shape: Shape | None = None,
        output_shape: Shape | None = None,
        implementation: str | None = None,
    ) -> Node:
        extra_attributes: dict[str, ShapeTuple | str] = {}
        for key, item in (("input_shape", input_shape), ("output_shape", output_shape)):
            if item is not None:
                extra_attributes[key] = item.to_tuple()
        if implementation is not None:
            extra_attributes["implementation"] = implementation
        if type is not None:
            extra_attributes["type"] = type
        if len(extra_attributes) > 0:
            return NodeImpl(name, attributes | extra_attributes)
        return NodeImpl(name, attributes)

    def edge(
        self,
        src: str,
        dst: str,
        attributes: ir.AttributeMapping = ir.AttributeMapping(),
        /,
        src_dst_indices: Iterable[tuple[int, int]] | tuple[str, str] = tuple(),
    ) -> Edge:
        if (
            isinstance(src_dst_indices, tuple)
            and len(src_dst_indices) > 0
            and isinstance(src_dst_indices[0], str)
        ):
            indices = src_dst_indices
        else:
            indices = tuple(cast(Iterable[tuple[int, int]], src_dst_indices))

        if len(indices) > 0:
            attributes = attributes | dict(src_dst_indices=indices)
        return EdgeImpl(src, dst, attributes)

    def graph(
        self,
        attributes: ir.AttributeMapping = ir.AttributeMapping(),
        *,
        type: str | None = None,
        name: str | None = None,
        other: ir.DataGraph[ir.Node, ir.Edge] | None = None,
    ) -> DataGraph:
        if other is not None:
            _graph = other.graph
            attributes = other.attributes | attributes
            node_attributes = other.node_attributes
        else:
            node_attributes = ir.AttributeMapping()
            _graph = ir.GraphImpl(lambda: ir.AttributeMapping())
        if type is not None:
            attributes = attributes.new_with(type=type)
        if name is not None:
            attributes = attributes.new_with(name=name)

        return DataGraphImpl(self, attributes, _graph, node_attributes)


def _check_and_get_name_fn(name: str | None, fn: Callable) -> str:
    if name is None:
        if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
            return fn.__name__
        else:
            raise Exception(f"you need to specify name explicitly for {fn}")
    return name


type Code = tuple[str, Sequence[str]]
type Registry = ir.Registry[DataGraph]
type TypeHandler = Callable[[DataGraph, Registry], Iterable[Code]]
type NonIterableTypeHandler = Callable[[DataGraph, Registry], Code]
