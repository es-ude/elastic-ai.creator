from collections.abc import Iterator, Sequence
from typing import Protocol

from elasticai.creator.vhdl.design.design import Design


class SequentialSubmodule(Protocol):
    def create_design(self, name: str) -> Design: ...


class SequentialDesignFactory(Protocol):
    def create_sequential_design(
        self, sub_designs: list[Design], name: str
    ) -> Design: ...


class SequentialDesignBuilder:
    def __init__(self, design_factory: SequentialDesignFactory) -> None:
        self._design_factory = design_factory

    def build(self, submodules: Sequence[SequentialSubmodule], name: str) -> Design:
        registry = _Registry()
        for module in submodules:
            registry.register(module.__class__.__name__.lower(), module)
        subdesigns = list(registry.build_designs())
        return self._design_factory.create_sequential_design(
            sub_designs=subdesigns, name=name
        )


class _Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, SequentialSubmodule] = {}
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

    def register(self, name: str, module: SequentialSubmodule):
        unique_name = self._make_name_unique(name)
        self._nodes[unique_name] = module
        self._increment_name_counter(name)

    def build_designs(self) -> Iterator[Design]:
        for name, module in self._nodes.items():
            yield module.create_design(name)
