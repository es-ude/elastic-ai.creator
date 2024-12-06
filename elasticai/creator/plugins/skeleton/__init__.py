__all__ = [
    "write_new_skeleton_pkg_to_build_dir",
    "compute_skeleton_id_hash",
    "MetaFile",
]

from .src.skeleton_pkg import (
    MetaFile,
    compute_skeleton_id_hash,
    write_new_skeleton_pkg_to_build_dir,
)

META = dict(
    version="0.1",
    static_components=("skeleton", "skeleton_pkg"),
)
