from typing import ContextManager, Protocol

from creator.file_generation.template import Template, TemplateExpander


class File(ContextManager, Protocol):
    def write(self, text: str):
        ...

    def close(self):
        ...


class Path(Protocol):
    def open(self) -> File:
        ...

    def create_subpath(self, name) -> "Path":
        ...


class TemplateWriter:
    def __init__(self, file: File):
        self._file = file

    def write(self, content: Template):
        expander = TemplateExpander(content)
        for line in expander.lines():
            self._file.write(line)
            self._file.write("\n")
