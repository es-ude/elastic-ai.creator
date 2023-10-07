import os
from pathlib import Path as _PyPath
from typing import Iterable

from .savable import File, Path
from .template import Template, TemplateExpander


class OnDiskFile(File):
    def __init__(self, full_path: str) -> None:
        self._full_path = full_path

    def write_text(self, text: Iterable[str]):
        self._create_parent_folder()
        path = _PyPath(self._full_path)
        with open(path, "w") as f:
            f.writelines((f"{line}\n" for line in text))

    def write(self, template: Template) -> None:
        expander = TemplateExpander(template)
        unfilled_variables = expander.unfilled_variables()
        if len(unfilled_variables) > 0:
            raise KeyError(
                "Template is not filled completely. The following variables are"
                f" unfilled: {', '.join(unfilled_variables)}."
            )
        self.write_text(expander.lines())

    def _create_parent_folder(self):
        full_path = _PyPath(self._full_path)
        folder = full_path.parent
        if not folder.exists():
            os.makedirs(folder)


class OnDiskPath(Path):
    def __init__(self, name: str, parent: str = ".") -> None:
        self._full_path = f"{parent}/{name}"

    def create_subpath(self, name: str) -> "OnDiskPath":
        return OnDiskPath(name, parent=self._full_path)

    def as_file(self, suffix: str) -> OnDiskFile:
        return OnDiskFile(full_path=f"{self._full_path}{suffix}")
