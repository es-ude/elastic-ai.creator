from collections import namedtuple
from collections.abc import Iterable
from re import Match

from elasticai.creator.ir import Graph, Lowerable, LoweringPass, Node
from .template import (
    Template,
    TemplateBuilder,
    AnalysingTemplateParameterType,
    TemplateParameterType,
)


class VhdlNode(Node):
    entity: str


class VhdlEntityIr(Graph, Lowerable):
    def __init__(self, name: str, type: str, generics: dict[str, str]) -> None:
        super().__init__()
        self._name = name
        self._type = type
        self._generics = dict((k, v) for k, v in generics.items())

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def generics(self) -> dict[str, str]:
        return self._generics


SourceFile = namedtuple("SourceFile", ("name", "code"))


class Vhdl2Ir(LoweringPass[VhdlEntityIr, tuple[str, Iterable[str]]]):
    pass
