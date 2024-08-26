from pathlib import Path

from elasticai.creator.file_generation.v2.savable import Savable
from elasticai.creator.test_utils.temporary_file_structure import (
    get_savable_file_structure,
)


class EmptySavable(Savable):
    def save_to(self, destination: Path) -> None:
        pass


class SingleFileSavable(Savable):
    def save_to(self, destination: Path) -> None:
        (destination / "a.txt").write_text("Hello\nWorld")


class NestedFileSavable(Savable):
    def save_to(self, destination: Path) -> None:
        (destination / "a.txt").write_text("/a.txt")
        (destination / "d1").mkdir()
        (destination / "d1" / "b.txt").write_text("/d1/b.txt")
        (destination / "d1" / "d2").mkdir()
        (destination / "d1" / "d2" / "c.txt").write_text("/d1/d2/c.txt")


def test_savable_without_files_leads_to_empty_file_structure() -> None:
    files = get_savable_file_structure(EmptySavable())
    assert dict() == files


def test_savable_with_single_file() -> None:
    files = get_savable_file_structure(SingleFileSavable())
    assert files == {"a.txt": "Hello\nWorld"}


def test_savable_with_nested_file_structure() -> None:
    files = get_savable_file_structure(NestedFileSavable())
    assert files == {
        "d1": {
            "d2": {
                "c.txt": "/d1/d2/c.txt",
            },
            "b.txt": "/d1/b.txt",
        },
        "a.txt": "/a.txt",
    }
