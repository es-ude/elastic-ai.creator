from dataclasses import dataclass
from typing import Iterator


@dataclass
class Port:
    inputs: dict[str, str]
    outputs: dict[str, str]

    def signals(self) -> Iterator[str]:
        for s, type in self.inputs.items():
            yield f"signal {s} : in {type}"
        for s, type in self.outputs.items():
            yield f"signal {s} : out {type}"


class VHDLEntity:
    def __init__(self, name: str, port: Port, generics: dict[str, str]):
        self._name = name
        self._generics = generics
        self._port = port

    def _generate_generic(self):
        if len(self._generics) > 0:
            yield "generic ("
            generics = tuple(self._generics.items())
            for generic, type in generics[:-1]:
                yield f"{generic} : {type};"
            generic, type = generics[-1]
            yield f"{generic} : {type}"
            yield ");"

    def _generate_port(self):
        yield "port ("
        signals = tuple(self._port.signals())
        for signal in signals[:-1]:
            yield f"{signal};"
        yield signals[-1]
        yield ");"

    def _generate_library_clause(self):
        yield from ("library ieee;", "use ieee.std_logic_1164.all;")

    def generate_entity(self) -> Iterator[str]:
        yield from self._generate_library_clause()
        yield f"entity {self._name} is"
        yield from self._generate_generic()
        yield from self._generate_port()
        yield "end entity;"
