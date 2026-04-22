"""Provides ways to use and extend the elasticai.creator plugin system.

The plugin systems evolves around the

* [`PluginLoader`](#elasticai.creator.plugin.PluginLoader) class
* [`PluginSymbol`](#elasticai.creator.plugin.PluginSymbol)protocol
* `meta.toml` files that describe the plugins

For convenience many functions convert the dicts loaded from the `meta.toml` file into `PluginSpec` objects.
The `meta.toml` file needs to define the value of each field of the `PluginSpec` class.
The function [`read_plugin_dicts_from_package`](#elasticai.creator.plugin.read_plugin_dicts_from_package) will read all plugins from the `plugins` key in the `meta.toml` file of a package.

**Example of a minimal `meta.toml` file**
```toml
[[plugins]]
name = "minimal_plugin"
target_platform = "elastic-node-v5"
target_runtime = "vhdl"
version = "0.1"
api_version = "0.1"
```

The few minimal fields that a plugin is required to define shall allow
plugin loaders to decide how to treat the plugin this could mean to

* ignore the plugin
* forward it to another software component
* check if the plugin is not compatible with the current setup

The following table lists these required fields:

:::{list-table}
* - Field name
  - Type
  - Description
* - **name**
  - `str`
  - The name of the plugin, used to identify the plugin
* - **target platform**
  - `str`
  - A string describing the target platform for the plugin, ie.
    the lowering pass it should be loaded into.
    Currently there is no strict definition of the semantics of this string.
* - **target runtime**
  - `str`
  - A string the runtime context for the plugin.
    Currently there is no strict definition of the semantics of this string.
* - **version**
  - `str`
  - A version string in the form `major.minor.[patch]`.
    Specifies the version of the plugin, ie. if you introduce a new feature or fix a bug, you should usually increase the minor version.
* - **api_version**
  - `str`
  - The version of the plugin API (plugin system) that this plugin was developed against.
    This is used to check if the plugin is compatible with the current system.
:::


:::{warning}
The set of required fields and their semantics is experimental and likely to change in the future.
:::

The `PluginLoader` will read that description from the `meta.toml` file
in a given package and use a user provided function to decide which
symbols to load from which module.
Assuming that each of these symbols implements the `PluginSymbol` protocol,
it will then call `load_into` on each of these symbols with a `plugin_receiver`.
The `plugin_receiver` is provided by the user as well.

Most other classes defined in this module are supposed to increase usability and expressiveness.

"""

import importlib.resources as _res
from collections.abc import Iterable

import tomlkit as toml

from .plugin_spec import PluginMap


def read_plugin_dicts_from_package(package: str) -> Iterable[PluginMap]:
    """read the meta.toml file from the package returning the list of plugin dictionaries."""
    t = _res.files(package).joinpath("meta.toml")
    parsed: list[dict] = []
    if t.is_file():
        with t.open("rb") as f:
            content = toml.load(f).unwrap()
            parsed.extend(content["plugins"])
    for d in parsed:
        d.update(dict(package=package))
    return parsed
