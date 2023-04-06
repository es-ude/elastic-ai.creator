"""
This module provides a parser for design_meta.toml files.

this is an example of such a meta file:

```toml
[elasticai.creator]
version = "==0.34"

[elasticai.creator.design]
protocol = "hw_block"
```

With the `version` you need to specify which versions of the elastic-ai creator your
`layer.py` is compatible with.
With `protocol` you define your supported protocol(s).
Additional protocol specific configuration needs to be defined under, e.g., `[elasticai.creator.design.protocol.hw_block]`.
Eg.
```toml
[elasticai.creator.design.protocols.hw_block]
pass_through = ["x_address", "y_address", "done"]
```
to specify that the generated base template should implement passing through the signals `x_address`, `y_address` and `done`.
TODO: how would we allow designers to specify more than one protocol?
TODO: rename design_meta.toml to layer_meta.toml?
TODO: how about we only require a folder containing a `layer.toml`. That file specifies the entry point for loading/using a layer.
"""

from pathlib import Path
import tomllib
from dataclasses import dataclass


@dataclass
class DesignMeta:
    version: str
    protocol: str

    @staticmethod
    def load(path: Path) -> "DesignMeta":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        data = data["elasticai"]["creator"]
        return DesignMeta(version=data["version"], protocol=data["design"]["protocol"])