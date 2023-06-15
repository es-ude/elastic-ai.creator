import os
from pathlib import Path as _PyPath

from elasticai.creator.vhdl.code_generation.template import Template, TemplateExpander
from elasticai.creator.vhdl.savable import File, Path


class OnDiskFile(File):
    def __init__(self, full_path: str) -> None:
        self._full_path = full_path

    def write(self, template: Template) -> None:
        expander = TemplateExpander(template)
        unfilled_variables = expander.unfilled_variables()
        if len(unfilled_variables) > 0:
            raise KeyError(
                "Template is not filled completly. The following variables are"
                f" unfilled: {', '.join(unfilled_variables)}."
            )
        full_path = _PyPath(self._full_path)
        folder = full_path.parent
        if not folder.exists():
            os.makedirs(folder)
        with open(full_path, "w") as f:
            f.writelines((f"{line}\n" for line in expander.lines()))


class OnDiskPath(Path):
    def __init__(self, name: str, parent: str = ".") -> None:
        self._full_path = f"{parent}/{name}"

    def create_subpath(self, name: str) -> "OnDiskPath":
        return OnDiskPath(name, parent=self._full_path)

    def as_file(self, suffix: str) -> OnDiskFile:
        return OnDiskFile(full_path=f"{self._full_path}{suffix}")
