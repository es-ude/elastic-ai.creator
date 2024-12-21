"""
The skeleton id should be computed and set after all other code is generated.
"""

from platform import python_version_tuple
import logging
from collections.abc import Iterable
from hashlib import blake2b, file_digest
from pathlib import Path


def compute_skeleton_id_hash(files: Iterable[Path]) -> bytes:
    logger = logging.getLogger(__name__)
    hash = _SkeletonIdHash()

    for vhd_file in files:
        logger.debug(f"hashing {vhd_file.as_uri}")
        hash.update(vhd_file)
    digest = hash.digest()
    logger.debug(f"raw_digest is {digest.hex()}")

    return digest


def replace_id_in_vhdl(code: Iterable[str], id: bytes) -> Iterable[str]:
    """
    Look for a line that starts with `constant SKELETON_ID` and replace it with
    the given id.
    """

    def split_hex(hex: str):
        for i in range(int(len(hex) / 2)):
            yield hex[i : i + 2]

    digest_as_hex_array = tuple(split_hex(id.hex()))

    for line in code:
        if _is_id(line):
            yield _build_skeleton_id_line(digest_as_hex_array)
            yield "\n"
        else:
            yield line


def update_skeleton_id_in_build_dir(build_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.debug("updating skeleton id")
    skeleton_pkg = None

    def is_not_skeleton_pkg(f: Path) -> bool:
        nonlocal skeleton_pkg
        if f.name == "skeleton_pkg.vhd":
            skeleton_pkg = f
            return False
        return True

    files_to_hash = filter(is_not_skeleton_pkg, build_dir.glob("*/**"))
    id = compute_skeleton_id_hash(files_to_hash)
    logger.debug(f"computed id is {id!r}")
    if skeleton_pkg is None:
        raise IOError("skeleton_pkg.vhd not found in build folder")
    with open(skeleton_pkg, "r") as f:
        code: Iterable[str] = f.readlines()
    logger.debug("updating skeleton_pkg.vhd")
    code = replace_id_in_vhdl(code, id)
    with open(skeleton_pkg, "w") as f:
        for line in code:
            f.write(line)
            f.write("\n")
    logger.debug("done")


def _is_id(line: str):
    return line.strip().startswith("constant SKELETON_ID")


def _build_skeleton_id_line(id: Iterable[str]) -> str:
    id_str = ", ".join(f'x"{b.upper()}"' for b in id)

    return f"  constant SKELETON_ID : skeleton_id_t := ({id_str});"


class _SkeletonIdHash:
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
