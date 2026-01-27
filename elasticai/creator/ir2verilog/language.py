from typing import Self
from typing import Iterator
from abc import abstractmethod
from abc import ABC

from elasticai.creator.hdl_ir import Node


class BaseWire(ABC):
    types: set[type["BaseWire"]] = set()

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def define(self) -> Iterator[str]: ...

    @classmethod
    def from_code(cls, code: str) -> "BaseWire":
        for t in cls.types:
            if t.can_create_from_code(code):
                return t.from_code(code)
        return NullWire.from_code(code)

    @classmethod
    @abstractmethod
    def can_create_from_code(cls, code: str) -> bool: ...

    @classmethod
    def register_type(cls, t: type["BaseWire"]) -> None:
        cls.types.add(t)

    @abstractmethod
    def make_instance_specific(self, instance: str) -> "BaseWire": ...


class NullWire(BaseWire):
    def __init__(self, name):
        self._name = name

    def define(self) -> Iterator[str]:
        yield from []

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return False

    @classmethod
    def from_code(cls, code: str) -> BaseWire:
        return cls("<unknown>")

    def make_instance_specific(self, instance: str) -> BaseWire:
        return self


class Wire(BaseWire):
    def define(self) -> Iterator[str]:
        yield f"wire {self.name};"

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return False

    def make_instance_specific(self, instance) -> Self:
        return type(self)(f"{self.name}_{instance}")


class VectorWire(BaseWire):
    def __init__(self, name: str, width: int):
        super().__init__(name)
        self._width = width

    @property
    def width(self) -> int:
        return self._width

    def define(self) -> Iterator[str]:
        yield f"wire [{self.width - 1}:0] {self.name};"

    @classmethod
    def can_create_from_code(cls, code: str) -> bool:
        return False

    def make_instance_specific(self, instance: str) -> Self:
        return type(self)(f"{self.name}_{instance}", self._width)


class Instance:
    """Represents a Verilog module instance.

    Aggregates all knowledge necessary to instantiate and use
    a Verilog module programmatically when generating code.
    """

    def __init__(
        self,
        node: Node,
        parameters: dict[str, str],
        ports: dict[str, BaseWire],
    ) -> None:
        self._node = node
        self._parameters = {k.upper(): v for k, v in parameters.items()}
        self._ports = {k: v.make_instance_specific(node.name) for k, v in ports.items()}

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def implementation(self) -> str:
        return self._node.implementation

    def define_signals(self) -> Iterator[str]:
        """Generate wire definitions for all ports."""
        for wire in self._ports.values():
            yield from wire.define()

    def instantiate(self) -> Iterator[str]:
        """Generate Verilog module instantiation code."""
        # Generate parameter list if any
        params = tuple(self._parameters.items())
        if params:
            yield f"{self.implementation} #("
            for i, (key, value) in enumerate(params):
                if i < len(params) - 1:
                    yield f"    .{key}({value}),"
                else:
                    yield f"    .{key}({value})"
            yield f") {self.name} ("
        else:
            yield f"{self.implementation} {self.name} ("

        # Generate port connections
        port_list = tuple(self._ports.items())
        for i, (port_name, wire) in enumerate(port_list):
            if i < len(port_list) - 1:
                yield f"    .{port_name}({wire.name}),"
            else:
                yield f"    .{port_name}({wire.name})"
        yield ");"
