from collections.abc import Callable

import pytest

from elasticai.creator.ir import Lowerable, LoweringPass
from elasticai.creator.lowering_pass_plugin import Loader, Plugin, SubFolderStructure
from elasticai.creator.plugin import read_plugins_from_package


@pytest.fixture
def plugin() -> Plugin:
    p = read_plugins_from_package("tests.integration_tests.minimal_plugin", Plugin)
    return p[0]


class PluginLoader(Loader):
    folders = SubFolderStructure("src", "vhdl", "static")


def test_can_read_plugin(plugin) -> None:
    p = plugin
    assert (
        Plugin(
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
    loader = PluginLoader(lower)
    lowerable = make_lowerable(["some", "important", "information"])
    loader.load_from_package("tests.integration_tests.minimal_plugin")
    assert ("some_important_information",) == tuple(lower((lowerable,)))
