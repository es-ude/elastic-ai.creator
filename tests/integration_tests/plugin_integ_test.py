from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, override

import pytest

import elasticai.creator.plugin as pl
from elasticai.creator.ir import Lowerable, LoweringPass
from elasticai.creator.plugin import PluginLoaderBase


@dataclass
class PluginSpec(pl.PluginSpec):
    generated: tuple[str, ...]


class PluginSymbol(Protocol):
    """Protocol for plugin symbols that can be loaded."""

    pass


class PluginLoader(PluginLoaderBase):
    """PluginLoader for test lowering passes."""

    def __init__(self, lowering: LoweringPass, target_runtime: str = "vhdl"):
        self._receiver = lowering
        self._target_runtime = target_runtime
        super().__init__(PluginSpec)

    @override
    def filter_plugin_dicts(
        self, plugins: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for p in plugins:
            if p["target_runtime"] == self._target_runtime:
                yield p

    @override
    def get_symbols(self, specs: Iterable[PluginSpec]) -> Iterable[PluginSymbol]:
        for spec in specs:
            yield from pl.import_symbols(spec.package, spec.generated)

    @override
    def load_symbol(self, symbol: PluginSymbol) -> None:
        # Check for target-specific load method
        load_method = f"load_{self._target_runtime}"
        if hasattr(symbol, load_method):
            getattr(symbol, load_method)(self._receiver)
        elif hasattr(symbol, "load_minimal"):
            symbol.load_minimal(self._receiver)
        else:
            raise TypeError(
                f"Failed to load plugin symbol: no {load_method} or load_minimal method found"
            )


class DummyLowerable(Lowerable):
    def __init__(self, type: str, more_data: list[str]):
        self._type = type
        self.more_data: list[str] = more_data

    @property
    def type(self) -> str:
        return self._type


@pytest.fixture
def plugin() -> PluginSpec:
    loader = PluginLoader(LoweringPass(), target_runtime="vhdl")
    specs = list(loader.get_specs("tests.integration_tests.minimal_plugin"))
    return specs[0]


def test_can_read_plugin(plugin: PluginSpec) -> None:
    assert plugin.name == "minimal_plugin"
    assert plugin.target_platform == "elastic-node-v5"
    assert plugin.target_runtime == "vhdl"
    assert plugin.version == "0.1"
    assert plugin.api_version == "0.1"
    assert plugin.generated == ["convolution"]
    assert plugin.package == "tests.integration_tests.minimal_plugin"


@pytest.fixture
def make_lowerable(plugin) -> Callable[[list[str]], DummyLowerable]:
    def dummy(more_data: list[str]) -> DummyLowerable:
        return DummyLowerable(plugin.generated[0], more_data)

    return dummy


def test_can_load_entire_plugin(
    make_lowerable: Callable[[list[str]], DummyLowerable],
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()
    loader = PluginLoader(lower, target_runtime="vhdl")
    lowerable = make_lowerable(["some", "important", "information"])
    loader.load_from_package("tests.integration_tests.minimal_plugin")

    assert ("some_important_information",) == tuple(lower((lowerable,)))


def test_can_load_lowering_pass_plugin(
    make_lowerable: Callable[[list[str]], DummyLowerable],
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()
    loader = PluginLoader(lower, target_runtime="lowering_pass_test")
    lowerable = DummyLowerable(
        "lowering_pass_conv", ["some", "important", "information"]
    )
    loader.load_from_package("tests.integration_tests.minimal_plugin")

    assert ("some_important_information",) == tuple(lower((lowerable,)))
