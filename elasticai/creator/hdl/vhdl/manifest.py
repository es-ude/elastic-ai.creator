"""
This module provides a parser for manifest.toml manifest files.

this is an example of such a manifest file:

```toml
[elasticai.creator]
version = "==0.34"

```

With the `version` you need to specify which versions of the elastic-ai creator your
`layer.py` is compatible with.
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
