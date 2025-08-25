"""
The skeleton id should be computed and set after all other code is generated.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from warnings import warn

from elasticai.creator.hw_function_id import _HwFunctionIdHash
from elasticai.creator.vhdl.hw_function_id import _replace_id_in_vhdl


def compute_skeleton_id_hash(files: Iterable[Path]) -> bytes:
    warn(
        "this function will be deprecated soon, use `HwFunctionIdUpdater` instead.",
        category=DeprecationWarning,
    )
    logger = logging.getLogger(__name__)
    hash = _HwFunctionIdHash()

    for vhd_file in files:
        logger.debug(f"hashing {vhd_file.as_uri}")
        hash.update(vhd_file)
    digest = hash.digest()
    logger.debug(f"raw_digest is {digest.hex()}")

    return digest


def replace_id_in_vhdl(code: Iterable[str], id: bytes) -> Iterable[str]:
    warn(
        "this function will be deprecated soon, use `HwFunctionIdUpdater` instead.",
        category=DeprecationWarning,
    )
    """
    Look for a line that starts with `constant SKELETON_ID` and replace it with
    the given id.
    """

    return _replace_id_in_vhdl(code, id)


def update_skeleton_id_in_build_dir(build_dir: Path) -> bytes:
    warn(
        "this function will be deprecated soon, use `HwFunctionIdUpdater` instead.",
        category=DeprecationWarning,
    )
    """insert the id into the skeleton_pkg.vhd file under `build_dir`."""

    logger = logging.getLogger(__name__)
    logger.debug("updating skeleton id")
    skeleton_pkg = None

    def is_not_skeleton_pkg(f: Path) -> bool:
        nonlocal skeleton_pkg
        if f.name == "skeleton_pkg.vhd":
            skeleton_pkg = f
            return False
        return True

    def files_recursive():
        for f in build_dir.glob("**/*"):
            if f.is_file():
                yield f

    files_to_hash = filter(is_not_skeleton_pkg, files_recursive())
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
    return id
