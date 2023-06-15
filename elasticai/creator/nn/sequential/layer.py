from collections.abc import Iterator
from typing import cast

import torch

from elasticai.creator.vhdl.design_base.design import Design
from elasticai.creator.vhdl.translatable import Translatable

from .design import Sequential as _SequentialDesign


class Sequential(Translatable, torch.nn.Sequential):
    def __init__(self, *submodules: Translatable):
        super().__init__(*cast(tuple[torch.nn.Module, ...], submodules))

    def translate(self, name: str) -> Design:
        registry = _Registry()
        submodules: list[Translatable] = [
            cast(Translatable, m) for m in self.children()
        ]
        for module in submodules:
            registry.register(module.__class__.__name__.lower(), module)
        subdesigns = list(registry.build_designs())
        return _SequentialDesign(
            sub_designs=subdesigns,
            name=name,
        )


class _Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, Translatable] = {}
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

    def register(self, name: str, d: Translatable):
        unique_name = self._make_name_unique(name)
        self._nodes[unique_name] = d
        self._increment_name_counter(name)

    def build_designs(self) -> Iterator[Design]:
        for name, module in self._nodes.items():
            yield module.translate(name)
