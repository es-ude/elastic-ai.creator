from collections.abc import Iterable, Iterator
from pathlib import Path

from elasticai.creator.hw_function_id import HwFunctionIdUpdater as _generalUpdater


def _replace_id_in_vhdl(code: Iterable[str], id: bytes) -> Iterator[str]:
    def split_hex(hex: str):
        for i in range(int(len(hex) / 2)):
            yield hex[i : i + 2]

    def _is_id(line: str):
        return line.strip().startswith("constant SKELETON_ID")

    def _build_skeleton_id_line(id: Iterable[str]) -> str:
        id_str = ", ".join(f'x"{b.upper()}"' for b in id)

        return f"  constant SKELETON_ID : skeleton_id_t := ({id_str});"

    digest_as_hex_array = tuple(split_hex(id.hex()))

    for line in code:
        if _is_id(line):
            yield _build_skeleton_id_line(digest_as_hex_array)
            yield "\n"
        else:
            yield line


class HwFunctionIdUpdater:
    def __init__(
        self,
        build_dir: Path,
        target_file: str | Path,
    ):
        self._updater = _generalUpdater(build_dir, target_file, _replace_id_in_vhdl)

    def compute_id(self) -> None:
        self._updater.compute_id()

    def write_id(self) -> None:
        self._updater.write_id()

    @property
    def id(self) -> bytes:
        return self._updater.id
