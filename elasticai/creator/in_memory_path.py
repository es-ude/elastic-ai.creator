from typing import Optional

from elasticai.creator.vhdl.code_generation.template import Template, TemplateExpander
from elasticai.creator.vhdl.savable import File, Path


class InMemoryFile(File):
    def __init__(self, name: str) -> None:
        self.text: list[str] = []
        self.name = name

    def write(self, template: Template) -> None:
        expander = TemplateExpander(template)
        unfilled_variables = expander.unfilled_variables()
        if len(unfilled_variables) > 0:
            raise KeyError(
                "Template is not filled completly. The following variables are"
                f" unfilled: {', '.join(unfilled_variables)}."
            )
        for line in expander.lines():
            self.text.append(line)


class InMemoryPath(Path):
    def __init__(self, name: str, parent: Optional["InMemoryPath"]) -> None:
        self.name = name
        self.children: dict[str, InMemoryPath | InMemoryFile] = dict()
        self.parent = parent

    def as_file(self, suffix: str) -> InMemoryFile:
        file = InMemoryFile(f"{self.name}{suffix}")
        if len(self.children) > 0:
            raise ValueError(
                f"non empty path {self.name}, "
                f"present children: {', '.join(self.children)}"
            )
        if self.parent is not None:
            self.parent.children[self.name] = file
        return file

    def __getitem__(self, item: str) -> "InMemoryPath | InMemoryFile":
        return self.children[item]

    def create_subpath(self, subpath_name: str) -> "InMemoryPath":
        subpath = InMemoryPath(name=subpath_name, parent=self)
        self.children[subpath_name] = subpath
        return subpath
