from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.temporary_file_structure import get_file_structure

from .template import fill_template, save_template


@dataclass
class Template:
    parameters: dict[str, str | list[str]]
    content: list[str]


@pytest.fixture
def template() -> Template:
    return Template(
        parameters=dict(a="hello", b="world"), content=["1) ${a}", "2) ${b}"]
    )


@pytest.fixture
def expected_lines() -> list[str]:
    return ["1) hello", "2) world"]


def test_fill_template(template: Template, expected_lines: list[str]) -> None:
    lines = fill_template(template)
    assert expected_lines == lines


def test_save_template(template: Template, expected_lines: list[str]) -> None:
    files = get_file_structure(
        template,
        save_obj_fn=lambda tpl, dest: save_template(tpl, dest / "test.txt"),
        read_file_fn=_read_saved_template,
    )
    assert expected_lines == files["test.txt"]


def _read_saved_template(file: Path) -> list[str]:
    with file.open("r") as in_file:
        raw_lines = in_file.readlines()
    return [line.rstrip("\n") for line in raw_lines]
