from elasticai.creator.plugins import Plugin, read_plugins_from_package


def test_can_read_plugin() -> None:
    p = read_plugins_from_package("tests.integration_tests.minimal_plugin")
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
        )
        == p[0]
    )
