from io import StringIO, TextIOBase

from .template import SimpleTemplate
from .template_writer import TemplateWriter


class File:
    def __init__(self):
        self._content = None
        self._string_io = StringIO()

    def inspect(self) -> str:
        return self._content

    def __enter__(self):
        return self._string_io.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._content = self._string_io.getvalue()
        return self._string_io.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self._content = self._string_io.getvalue()
        self._string_io.close()

    def __str__(self):
        return f"File <{self._content}>"


class Path:
    def __init__(self, full_path: str):
        self.children = {}
        self.full_path = full_path
        self._content = File()

    def create_subpath(self, name: str) -> "Path":
        self.children[name] = Path(full_path="/".join([self.full_path, name]))
        return self.children[name]

    def inspect(self):
        return self._content.inspect()

    def open(self, mode: str = "w") -> TextIOBase:
        self._content = File()
        return self._content

    def __str__(self):
        return f"\t{self.full_path}\n\t" + "\n\t".join(
            [repr(child) for child in self.children]
        )


def test_template_writer():
    template = SimpleTemplate(parameters={}, content=["a", "b"])
    build_dir = Path("root")
    file_path = build_dir.create_subpath("myfile.txt")
    with file_path.open() as f:
        writer = TemplateWriter(f)
        writer.write(template)
    expected = "a\nb\n"
    assert expected == file_path.inspect()
