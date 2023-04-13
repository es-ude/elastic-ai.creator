"""
This module provides a parser for layer.toml manifest files.

this is an example of such a manifest file:

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
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elasticai.creator.hdl.vhdl.base_template_generator import BaseTemplateGenerator


@dataclass
class Manifest:
    version: str
    layer: dict

    @staticmethod
    def parse(text: str) -> "Manifest":
        data = tomllib.loads(text)
        return Manifest._create_manifest(data)

    @staticmethod
    def _create_manifest(data) -> "Manifest":
        data = Manifest._get_creator_data(data)
        permitted_pass_through_values = {"x", "y_address", "enable"}
        actual_pass_through_values = set(data["layer"]["pass_through"])
        print(actual_pass_through_values)
        print(permitted_pass_through_values.intersection(actual_pass_through_values))
        if (
            permitted_pass_through_values.intersection(actual_pass_through_values) == {}
            and actual_pass_through_values != {}
        ):
            raise InvalidManifestConfig
        return Manifest(version=data["version"], layer=data["layer"])

    @staticmethod
    def _get_creator_data(data):
        return data["elasticai"]["creator"]

    @staticmethod
    def load(path: Path) -> "Manifest":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return Manifest._create_manifest(data)


class InvalidManifestConfig(Exception):
    pass
