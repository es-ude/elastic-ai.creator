"""
The skeleton id should be computed and set after all other code is generated.
"""

import logging
from collections.abc import Iterable
from hashlib import blake2b, file_digest
from importlib.metadata import version as _pypackage_version
from itertools import filterfalse
from pathlib import Path

import tomlkit as toml
from tomlkit.toml_file import TOMLFile


def _is_id(line: str):
    return line.strip().startswith("constant SKELETON_ID")


def _build_skeleton_id_line(id: Iterable[str]) -> str:
    id_str = ", ".join(f'x"{b.upper()}"' for b in id)

    return f"  constant SKELETON_ID : skeleton_id_t := ({id_str});"


def replace_id_in_code(code: Iterable[str], id: Iterable[str]) -> Iterable[str]:
    """
    Look for a line that starts with `constant SKELETON_ID` and replace it with
    the given id.
    """
    for line in code:
        if _is_id(line):
            yield _build_skeleton_id_line(id)
            yield "\n"
        else:
            yield line


class SkeletonIdHash:
    SIZE: int = 16

    def __init__(self):
        self._digests = []

    def _hash(self):
        return blake2b(digest_size=self.SIZE)

    def update(self, lines: Iterable[str] | Path) -> None:
        h = self._hash()

        if isinstance(lines, Path):
            with open(lines, "rb") as f:
                self._digests.append(file_digest(f, self._hash).digest())
        else:
            for line in lines:
                h.update(line.encode())
            self._digests.append(h.digest())

    def digest(self) -> bytes:
        h = self._hash()
        for d in sorted(self._digests):
            h.update(d)
        return h.digest()


def compute_skeleton_id_hash(
    build_dir: Path, file_with_default_id: str = "skeleton_pkg.vhd"
) -> bytes:
    logger = logging.getLogger(__name__)
    hash = SkeletonIdHash()
    logger.info("writing skeleton id to skeleton_pkg.vhd")
    all_files = build_dir.glob("**/*.vhd")

    def skip_default_id_for_idempotency(file: Path) -> bool:
        return file.name == file_with_default_id

    files_to_hash = filterfalse(skip_default_id_for_idempotency, all_files)
    for vhd_file in files_to_hash:
        logger.debug(f"hashing {vhd_file.as_uri}")
        hash.update(vhd_file)
    digest = hash.digest()
    logger.debug(f"raw_digest is {digest.hex()}")

    return digest


def get_skeleton_pkg_file(build_dir: Path) -> Path:
    for file in build_dir.glob("**/*.vhd"):
        if file.name == "skeleton_pkg.vhd":
            return file
    raise IOError(f"could not find skeleton_pkg.vhd file in {build_dir}")


class MetaFile:
    version = "0.1"

    def __init__(self, data):
        self._data = data

    @classmethod
    def load(cls, build_dir: Path) -> "MetaFile":
        with open(build_dir / "meta.toml", "r") as f:
            data = toml.load(f)
        return cls(data)

    def set_skeleton_id(self, skeleton_id: bytes):
        self._data["skeleton_id"] = skeleton_id.hex()

    def save(self, build_dir: Path) -> None:
        f = TOMLFile(build_dir / "meta.toml")
        f.write(self._data)

    @classmethod
    def default(cls) -> "MetaFile":
        creator_version = _pypackage_version("elasticai.creator")
        data = dict(
            meta_version=cls.version,
            creator_version=creator_version,
            hw_accelerator_version="0.1",
            skeleton_id='"<invalid"  # this should be computed and set'
            " automatically",
            name="my_network",
            description="""An optional possibly very long description of what
                            your accelerator does and how to use it.""",
        )
        return cls(data)


def write_new_skeleton_pkg_to_build_dir(build_dir: Path, skeleton_id: bytes):
    logger = logging.getLogger(__name__)
    digest = skeleton_id
    skeleton_pkg = get_skeleton_pkg_file(build_dir)
    logger.debug(f"raw_digest is {digest.hex()}")

    def split_hex(hex: str):
        for i in range(int(len(hex) / 2)):
            yield hex[i : i + 2]

    digest_as_hex_array = tuple(split_hex(digest.hex()))
    with open(skeleton_pkg, "r") as f:
        logger.debug("reading {skeleton_pkg}")
        code = f.readlines()
    with open(skeleton_pkg, "w") as f:
        logger.debug("writing {skeleton_pkg}")
        f.writelines(replace_id_in_code(code, digest_as_hex_array))
    logger.info("done.")
