from collections.abc import Iterator
from typing import Protocol, cast

from torch.nn import Module as _torchModule
from torch.nn import Sequential as torchSequential

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.vhdl.designs.sequential import (
    Sequential as _SequentialDesign,
)

from .module import Module


class TranslatableModule(Module, Protocol):
    def translate_to_vhdl(self, name: str) -> Design:
        ...


# TODO: Implement abstract Translatable


class Sequential(torchSequential):
    def __init__(self, submodules: tuple[TranslatableModule, ...]):
        super().__init__(*cast(tuple[_torchModule], submodules))

    def translate(self) -> Design:
        registry = _Registry()
        submodules: list[TranslatableModule] = [
            cast(TranslatableModule, m) for m in self.children()
        ]
        for module in submodules:
            registry.register(module.__class__.__name__.lower(), module)
        subdesigns = list(registry.build_designs())
        x_address_width = 1
        y_address_width = 1
        if len(subdesigns) == 0:
            x_width = 1
            y_width = 1

        else:
            front = subdesigns[0]
            back = subdesigns[-1]
            x_width = front.port["x"].width
            y_width = back.port["y"].width
            found_y_address = False
            found_x_address = False
            for design in subdesigns:
                if "y_address" in design.port and not found_y_address:
                    found_y_address = True
                    y_address_width = design.port["y_address"].width
                if "x_address" in design.port and not found_x_address:
                    found_x_address = True
                    x_address_width = back.port["x_address"].width
        return _SequentialDesign(
            sub_designs=subdesigns,
            x_width=x_width,
            y_width=y_width,
            x_address_width=x_address_width,
            y_address_width=y_address_width,
        )


class TranslatableToDesign(Protocol):
    def translate(self) -> Design:
        ...


class _Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, TranslatableModule] = {}
        self._name_counters: dict[str, int] = {}

    def _make_name_unique(self, name: str) -> str:
        return f"{name}_{self._get_counter_for_name(name)}"

    def _get_counter_for_name(self, name: str) -> int:
        if name in self._name_counters:
            return self._name_counters[name]
        else:
            return 0

    def _increment_name_counter(self, name: str):
        self._name_counters[name] = 1 + self._get_counter_for_name(name)

    def register(self, name: str, d: TranslatableModule):
        unique_name = self._make_name_unique(name)
        self._nodes[unique_name] = d
        self._increment_name_counter(name)

    def build_designs(self) -> Iterator[Design]:
        for name, module in self._nodes.items():
            yield module.translate_to_vhdl(name)
