import tempfile
from pathlib import Path


class TemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(self) -> None:
        super().__init__(
            suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False, delete=True
        )

    def __enter__(self) -> Path:
        return Path(super().__enter__())


def read_lines_from_file(file: Path) -> list[str]:
    with file.open("r") as in_file:
        raw_lines = in_file.readlines()
    return [line.rstrip("\n") for line in raw_lines]
