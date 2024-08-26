import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from elasticai.creator.file_generation.savable import Savable

T = TypeVar("T")


class _TemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(self) -> None:
        super().__init__(
            suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False, delete=True
        )

    def __enter__(self) -> Path:
        return Path(super().__enter__())


def _load_file_structure_from_disc(
    directory: Path, read_file_fn: Callable[[Path], Any]
) -> dict[str, Any]:
    def load_files(directory: Path) -> dict[str, Any]:
        files = list(directory.glob("*"))
        structure = dict()
        for file in files:
            structure[file.name] = (
                load_files(file) if file.is_dir() else read_file_fn(file)
            )
        return structure

    return load_files(directory)


def get_file_structure(
    obj: T, save_obj_fn: Callable[[T, Path], None], read_file_fn: Callable[[Path], Any]
) -> dict[str, Any]:
    with _TemporaryDirectory() as destination:
        save_obj_fn(obj, destination)
        return _load_file_structure_from_disc(destination, read_file_fn)


def get_savable_file_structure(savable: Savable) -> dict[str, Any]:
    return get_file_structure(
        savable,
        save_obj_fn=lambda s, d: s.save_to(d),
        read_file_fn=lambda f: f.read_text(),
    )
