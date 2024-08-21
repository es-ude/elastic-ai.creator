from dataclasses import dataclass

import pytest

from .template import fill_template, save_template
from .temporary import TemporaryDirectory, read_lines_from_file


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
    with TemporaryDirectory() as out_dir:
        out_file = out_dir / "outputs.txt"
        save_template(template, out_file)
        written_lines = read_lines_from_file(out_file)
        assert expected_lines == written_lines
