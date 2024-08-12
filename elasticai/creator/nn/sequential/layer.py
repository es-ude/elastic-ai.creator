import abc
from abc import abstractmethod
from collections.abc import Iterator
from typing import cast

import torch

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design_creator import DesignCreator

from .design import Sequential as _SequentialDesign


class Sequential(DesignCreator, torch.nn.Sequential):
    def __init__(self, *submodules: DesignCreator):
        super().__init__(*cast(tuple[torch.nn.Module, ...], submodules))

    def create_design(self, name: str) -> Design:
        registry = _Registry()
        submodules: list[DesignCreator] = [
            cast(DesignCreator, m) for m in self.children()
        ]
        for module in submodules:
            registry.register(module.__class__.__name__.lower(), module)
        subdesigns = list(registry.build_designs())
        return _SequentialDesign(
            sub_designs=subdesigns,
            name=name,
        )


class IntForwardSubmission(torch.nn.Module, DesignCreator):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create_design(self, name: str) -> Design:
        ...

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def int_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...


class IntegerSequential(Sequential):
    def __init__(self, *submodules: IntForwardSubmission):
        super().__init__(*submodules)

        self.submodules = submodules

    def int_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert not self.training, "int_forward() should only be called in eval mode"

        x = inputs
        for submodule in self.submodules:
            x = submodule.int_forward(x)
        x = self.submodules[-1].output_QParams.dequantizeProcess(x)

        return x


class _Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, DesignCreator] = {}
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

    def register(self, name: str, d: DesignCreator):
        unique_name = self._make_name_unique(name)
        self._nodes[unique_name] = d
        self._increment_name_counter(name)

    def build_designs(self) -> Iterator[Design]:
        for name, module in self._nodes.items():
            yield module.create_design(name)
