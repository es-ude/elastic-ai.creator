import pytest

import elasticai.creator.plugin as p


def test_importing_plugin_with_unkown_fields_raises_meaningful_error():
    config_from_file = {
        "name": "my_plugin",
        "api_version": "0.1",
        "version": "0.2",
        "package": "mypackage",
        "target_platform": "env5",
        "target_runtime": "vhdl",
        "new_unknown_field": "value",
    }
    with pytest.raises(
        p.UnexpectedFieldError,
        match="unexpected fields {'new_unknown_field'} for plugin 'Plugin'",
    ):
        p.build_plugin(config_from_file, p.Plugin)
