"""
This module provides a parser for manifest.toml manifest files.

this is an example of such a manifest file:

```toml
[elasticai.creator]
version = "==0.34"

```

With the `version` you need to specify which versions of the elastic-ai creator your
`layer.py` is compatible with.
Eg.
```toml
[elasticai.creator.layer]
pass_through = ["x_address", "y_address", "enable"]
```
to specify that the generated base template should implement passing through the signals `x_address`, `y_address` and `enable`.
"""

import tomllib
from dataclasses import dataclass


def parse(text: str) -> "Manifest":
    data = tomllib.loads(text)
    data = data["elasticai"]["creator"]
    return Manifest(version=data["version"], layer=data["layer"])


@dataclass
class Manifest:
    version: str
    layer: dict
