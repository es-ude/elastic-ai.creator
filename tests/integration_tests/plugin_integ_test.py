import dataclasses
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from inspect import signature

import pytest

from elasticai.creator.ir import Lowerable, LoweringPass
from elasticai.creator.lowering_pass_plugin import (
    PluginLoader as LoweringPassPluginLoader,
)
from elasticai.creator.lowering_pass_plugin import (
    PluginSpec,
)
from elasticai.creator.plugin import (
    PluginDict,
    PluginLoader,
    build_plugin_spec,
    read_plugin_dicts_from_package,
)


@pytest.fixture
def plugin() -> PluginSpec:
    p = list(read_plugin_dicts_from_package("tests.integration_tests.minimal_plugin"))[
        0
    ]
    return build_plugin_spec(p, PluginSpec)


def test_can_read_plugin(plugin) -> None:
    p = plugin
    assert (
        PluginSpec(
            name="minimal_plugin",
            target_platform="elastic-node-v5",
            target_runtime="vhdl",
            version="0.1",
            api_version="0.1",
            generated=("convolution",),
            templates=("skeleton_id_pkg",),
            static_files=("constraints.xdc",),
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

    def extract_symbols(data: Iterable[PluginDict]) -> Iterator[tuple[str, set[str]]]:
        plugins = map(partial(build_plugin_spec, spec_type=PluginSpec), data)
        for p in plugins:
            print(p)
            if p.target_runtime != "vhdl":
                continue
            yield f"{p.package}.src", set(p.generated)

    loader = PluginLoader(extract_fn=extract_symbols, plugin_receiver=lower)
    lowerable = make_lowerable(["some", "important", "information"])
    loader.load_from_package("tests.integration_tests.minimal_plugin")
    print(lower.__dict__["_fns"].__dict__["_fns"])

    assert ("some_important_information",) == tuple(lower((lowerable,)))


def test_can_load_lowering_pass_plugin(
    make_lowerable: Callable[[list[str]], DummyLowerable],
) -> None:
    lower: LoweringPass[DummyLowerable, str] = LoweringPass()

    loader = LoweringPassPluginLoader(
        target_runtime="lowering_pass_test", lowering=lower
    )
    lowerable = DummyLowerable(
        "lowering_pass_conv", ["some", "important", "information"]
    )
    loader.load_from_package("tests.integration_tests.minimal_plugin")
    print(lower.__dict__["_fns"].__dict__["_fns"])

    assert ("some_important_information",) == tuple(lower((lowerable,)))
