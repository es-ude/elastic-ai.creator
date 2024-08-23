from abc import ABC, abstractmethod
from pathlib import Path

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from tests.design_file_structure import design_file_structure


class DesignForTesting(Design, ABC):
    def port(self) -> Port:
        raise NotImplemented()

    @abstractmethod
    def save_to(self, destination: Path) -> None:
        pass


class EmptyDesign(DesignForTesting):
    def save_to(self, destination: Path) -> None:
        pass


class SingleFileDesign(DesignForTesting):
    def save_to(self, destination: Path) -> None:
        (destination / "a.txt").write_text("Hello\nWorld")


class NestedFileDesign(DesignForTesting):
    def save_to(self, destination: Path) -> None:
        (destination / "a.txt").write_text("/a.txt")
        (destination / "d1").mkdir()
        (destination / "d1" / "b.txt").write_text("/d1/b.txt")
        (destination / "d1" / "d2").mkdir()
        (destination / "d1" / "d2" / "c.txt").write_text("/d1/d2/c.txt")


def test_design_without_files_leads_to_empty_file_structure() -> None:
    files = design_file_structure(EmptyDesign("test"))
    assert dict() == files


def test_design_with_single_file() -> None:
    files = design_file_structure(SingleFileDesign("test"))
    assert files == {"a.txt": "Hello\nWorld"}


def test_design_with_nested_file_structure() -> None:
    files = design_file_structure(NestedFileDesign("test"))
    assert files == {
        "d1": {
            "d2": {
                "c.txt": "/d1/d2/c.txt",
            },
            "b.txt": "/d1/b.txt",
        },
        "a.txt": "/a.txt",
    }
