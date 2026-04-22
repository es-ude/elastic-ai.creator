from importlib.metadata import version as _pypackage_version
from pathlib import Path

import tomlkit as toml
from tomlkit.toml_file import TOMLFile


class MetaFile:
    version = "0.1"

    def __init__(self, data):
        self._data = data

    @classmethod
    def load(cls, file: Path) -> "MetaFile":
        with open(file, "r") as f:
            data = toml.load(f)
        return cls(data)

    def set_skeleton_id(self, skeleton_id: bytes):
        self._data["skeleton_id"] = skeleton_id.hex()

    def save(self, file: Path) -> None:
        f = TOMLFile(file)
        f.write(self._data)

    @classmethod
    def default(cls) -> "MetaFile":
        creator_version = _pypackage_version("elasticai.creator")
        data = dict(
            meta_version=cls.version,
            creator_version=creator_version,
            hw_accelerator_version="0.1",
            skeleton_id='"<invalid>"  # this should be computed and set automatically',
            name="<your_network>",
            description="""An optional possibly very long description of what
                            your accelerator does and how to use it.""",
        )
        return cls(data)
