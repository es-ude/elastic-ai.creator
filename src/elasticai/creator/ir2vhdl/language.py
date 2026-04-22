import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

import elasticai.creator.function_dispatch as FD
from elasticai.creator.hdl_ir import Node, Shape


def _check_and_get_fn_name(name: str | None, fn: Callable) -> str:
    """Get function name for registration, either from parameter or function.__name__."""
    if name is None:
        if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
            return fn.__name__
        else:
            raise Exception(f"you need to specify name explicitly for {fn}")
    return name


class Signal(ABC):
    types: set[type["Signal"]] = set()

    @abstractmethod
    def define(self) -> Iterator[str]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        for t in cls.types:
            if t.can_create_from_code(code):
                return t.from_code(code)
        return NullDefinedLogicSignal.from_code(code)

    @classmethod
    @abstractmethod
    def can_create_from_code(cls, code: str) -> bool: ...

    @classmethod
    def register_type(cls, t: type["Signal"]) -> None:
        cls.types.add(t)

    @abstractmethod
    def make_instance_specific(self, instance: str) -> "Signal": ...


class LogicSignal(Signal):
    def __init__(self, name: str):
        self._name = name

    def define(self) -> Iterator[str]:
        yield f"signal {self._name} : std_logic := '0';"

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return cls._search(code) is not None

    @classmethod
    def _search(cls, code: str) -> re.Match[str] | None:
        match = re.search(
            r"signal ([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*std_logic(?:\s+|;)", code
        )
        return match

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        match = cls._search(code)
        if match is None:
            raise ValueError(f"Cannot create signal from code: {code}")
        (name,) = match.groups()
        return cls(name)

    def make_instance_specific(self, instance: str) -> Signal:
        return self.__class__(f"{self.name}_{instance}")

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if isinstance(other, LogicSignal):
            return self._name == other._name
        return False


class LogicVectorSignal(Signal):
    def __init__(self, name: str, width: int):
        self._name = name
        self._width = width

    def define(self) -> Iterator[str]:
        yield f"signal {self._name} : std_logic_vector({self._width} - 1 downto 0) := (others => '0');"

    @property
    def name(self) -> str:
        return self._name

    @property
    def width(self) -> int:
        return self._width

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return cls._search(code) is not None

    @classmethod
    def _search(cls, code: str) -> re.Match[str] | None:
        match = re.match(
            r"signal ([a-zA-Z_][a-zA-Z0-9_]*)\s*: std_logic_vector\((\d+|(?:\d+ - \d+)) downto 0\)",
            code,
        )
        return match

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        match = cls._search(code)
        if match is None:
            raise ValueError(f"Cannot create signal from code: {code}")
        name, width = match.groups()
        if " - " in width:
            a, b = width.split(" - ")
            width = str(int(a) - int(b))
        return cls(name, int(width) + 1)

    def make_instance_specific(self, instance: str) -> Signal:
        return self.__class__(f"{self.name}_{instance}", self.width)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if isinstance(other, LogicVectorSignal):
            return self._name == other._name and self._width == other._width
        return False


class NullDefinedLogicSignal(Signal):
    def __init__(self, name):
        self._name = name

    def define(self) -> Iterator[str]:
        yield from []

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return False

    @classmethod
    def from_code(cls, code: str) -> "Signal":
        return cls("<unknown>")

    def make_instance_specific(self, instance: str) -> Signal:
        return self


for t in (LogicSignal, LogicVectorSignal, NullDefinedLogicSignal):
    Signal.register_type(t)


class PortMap:
    def __init__(self, map: dict[str, Signal]):
        self._signals: dict[str, Signal] = map

    def as_dict(self) -> dict[str, str]:
        return {k: tuple(v.define())[0] for k, v in self._signals.items()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "PortMap":
        return cls({k: Signal.from_code(v) for k, v in data.items()})

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if isinstance(other, PortMap):
            return self._signals == other._signals
        return False


class Instance:
    """Represents an entity that we can/want instantiate.

    The aggregates all the knowledge that is necessary to
    instantiate and use the corresponding entity programmatically,
    when generating vhdl code.
    """

    def __init__(
        self,
        node: Node,
        generic_map: dict[str, str],
        port_map: dict[str, Signal],
    ):
        self._node = node
        self._generics: dict[str, str] = {k.lower(): v for k, v in generic_map.items()}
        self.port_map = {
            k: v.make_instance_specific(self._node.name) for k, v in port_map.items()
        }

    @property
    def input_shape(self) -> Shape:
        return self._node.input_shape

    @property
    def output_shape(self) -> Shape:
        return self._node.output_shape

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def implementation(self) -> str:
        return self._node.implementation

    def define_signals(self) -> Iterator[str]:
        for s in self.port_map.values():
            yield from s.define()

    def instantiate(self) -> Iterator[str]:
        yield from (f"{self.name}: entity work.{self.implementation}(rtl) ",)
        generics = tuple(self._generics.items())
        if len(generics) > 0:
            yield "generic map ("
            for key, value in generics[:-1]:
                yield f"  {key.upper()} => {value},"
            for g in generics[-1:]:
                yield f"  {g[0].upper()} => {g[1]}"
            yield "  )"
        port_map = tuple(self.port_map.items())
        yield "  port map ("

        for k, v in port_map[:-1]:
            yield f"    {k} => {v.name},"
        for k, v in port_map[-1:]:
            yield f"    {k} => {v.name}"

        yield "  );"


class InstanceFactory:
    """Automatically creates Instances from VhdlNodes based on their `type` field."""

    @FD.dispatch_method(str)
    def _handle_type(self, fn: Callable[[Node], Instance], node: Node) -> Instance:
        return fn(node)

    @_handle_type.key_from_args
    def _get_type_from_node(self, node: Node) -> str:
        return node.type

    @FD.registrar_method
    def register(
        self,
        type: str | None,
        fn: Callable[[Node], Instance],
    ) -> Callable[[Node], Instance]:
        type = _check_and_get_name_fn(type, fn)
        return self._handle_type.register(type, fn)

    def __call__(self, node: Node) -> Instance:
        return self._handle_type(node)


def _check_and_get_name_fn(name: str | None, fn: Callable) -> str:
    if name is None:
        if hasattr(fn, "__name__") and isinstance(fn.__name__, str):
            return fn.__name__
        else:
            raise Exception(f"you need to specify name explicitly for {fn}")
    return name
