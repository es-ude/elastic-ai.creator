import pathlib
from itertools import chain
from typing import Iterable

from elasticai.creator.vhdl.language import (
    ContextClause,
    LibraryClause,
    UseClause,
    Entity,
    InterfaceVariable,
    DataType,
    Mode,
    Architecture,
    InterfaceSignal,
)


class MacAsync:
    def __init__(self, component_name, data_width, frac_width):
        self.component_name = component_name
        self.data_width = data_width
        self.frac_width = frac_width

    def __call__(self) -> Iterable[str]:
        with open(pathlib.Path(__file__).parent.resolve().joinpath("templates/mac_async.tpl.vhd"), "r") as f:
            template = f.read()

        source_code = template.format(entity_name=self.component_name, data_width=self.data_width, frac_width=self.frac_width)
        yield from map(lambda s: s.strip(" "), source_code.splitlines())