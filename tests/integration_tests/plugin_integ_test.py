from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest

import elasticai.creator.plugin as pl
from elasticai.creator.ir import Lowerable, LoweringPass


@dataclass
class PluginSpec(pl.PluginSpec):
    generated: list[str]


@pytest.fixture
def plugin() -> PluginSpec:
    p = list(
        pl.read_plugin_dicts_from_package("tests.integration_tests.minimal_plugin")
    )[0]
    return pl.build_plugin_spec(p, PluginSpec)


def test_can_read_plugin(plugin) -> None:
    p = plugin
    assert (
        PluginSpec(
            name="minimal_plugin",
            target_platform="elastic-node-v5",
            target_runtime="vhdl",
            version="0.1",
            api_version="0.1",
            generated=[
                "convolution",
            ],
            package="tests.integration_tests.minimal_plugin",
        )
        == p
    )


class DummyLowerable(Lowerable):
    def __init__(self, type: str, more_data: list[str]):
        self._type = type
        self.more_data: list[str] = more_data

    @property
    def type(self) -> str:
        return self._type


@pytest.fixture
def make_lowerable(plugin) -> Callable[[list[str]], DummyLowerable]:
    def dummy(more_data: list[str]) -> DummyLowerable:
        return DummyLowerable(plugin.generated[0], more_data)

    return dummy


def test_can_load_entire_plugin(
    make_lowerable: Callable[[list[str]], DummyLowerable],
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()

    def extract_symbols(p: PluginSpec) -> Iterator[pl.PluginSymbol]:
        if p.target_runtime == "vhdl":
            yield from pl.import_symbols(f"{p.package}.src", set(p.generated))

    loader = pl.PluginLoader(
        fetch=pl.SymbolFetcherBuilder(PluginSpec).add_fn(extract_symbols).build(),
        plugin_receiver=lower,
    )
    lowerable = make_lowerable(["some", "important", "information"])
    loader.load_from_package("tests.integration_tests.minimal_plugin")

    assert ("some_important_information",) == tuple(lower((lowerable,)))


def test_can_load_lowering_pass_plugin(
    make_lowerable: Callable[[list[str]], DummyLowerable],
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()

    def fetch(p: PluginSpec) -> Iterator[pl.PluginSymbol]:
        if p.target_runtime == "lowering_pass_test":
            yield from pl.import_symbols(f"{p.package}.src", set(p.generated))

    loader = pl.PluginLoader(
        fetch=pl.SymbolFetcherBuilder(PluginSpec).add_fn(fetch).build(),
        plugin_receiver=lower,
    )
    lowerable = DummyLowerable(
        "lowering_pass_conv", ["some", "important", "information"]
    )
    loader.load_from_package("tests.integration_tests.minimal_plugin")
    print(lower.__dict__["_fns"].__dict__["_fns"])

    assert ("some_important_information",) == tuple(lower((lowerable,)))
