from collections.abc import Iterator
from typing import cast

import torch

from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.vhdl.design.design import Design

from .design import Sequential as _SequentialDesign


class Sequential(DesignCreatorModule, torch.nn.Sequential):
    def __init__(self, *submodules: DesignCreatorModule):
        super().__init__(*submodules)

    def create_design(self, name: str) -> Design:
        registry = _Registry()
        submodules = [cast(DesignCreatorModule, m) for m in self.children()]
        for module in submodules:
            registry.register(module.__class__.__name__.lower(), module)
        subdesigns = list(registry.build_designs())
        return _SequentialDesign(
            sub_designs=subdesigns,
            name=name,
        )


class _Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, DesignCreatorModule] = {}
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

    def register(self, name: str, d: DesignCreatorModule):
        unique_name = self._make_name_unique(name)
        self._nodes[unique_name] = d
        self._increment_name_counter(name)

    def build_designs(self) -> Iterator[Design]:
        for name, module in self._nodes.items():
            yield module.create_design(name)
