from io import StringIO, TextIOBase


class VirtualFile:
    def __init__(self):
        self._content = None
        self._string_io = StringIO()

    def read(self) -> str:
        return self._content

    def __enter__(self):
        return self

    def write(self, text: str):
        self._string_io.write(text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._content = self._string_io.getvalue()
        return self._string_io.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self._content = self._string_io.getvalue()
        self._string_io.close()

    def __str__(self):
        return f"File <{self._content}>"


class VirtualPath:
    def __init__(self, full_path: str):
        self.children = {}
        self.full_path = full_path
        self._content = VirtualFile()

    def create_subpath(self, name: str) -> "VirtualPath":
        self.children[name] = VirtualPath(full_path="/".join([self.full_path, name]))
        return self.children[name]

    def read(self):
        return self._content.read()

    def open(self, mode: str = "w") -> TextIOBase:
        self._content = VirtualFile()
        return self._content

    def __str__(self):
        return f"\t{self.full_path}\n\t" + "\n\t".join(
            [repr(child) for child in self.children]
        )
