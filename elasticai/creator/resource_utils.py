from contextlib import contextmanager
from importlib import resources
from pathlib import PurePath
from typing import AnyStr, Iterable, TextIO

PathType = str | PurePath
Package = resources.Package


@contextmanager
def get_file(package: Package, file_name: str) -> TextIO:
    file_not_found = True
    for resource in resources.files(package).iterdir():
        if resource.name == file_name:
            file_not_found = False
            with resources.as_file(resource) as file:
                with open(file, "r") as opened_file:
                    yield opened_file
    if file_not_found:
        raise FileNotFoundError(
            f"The file '{file_name}' in package '{package}' does not exist."
        )


def read_text(
    package: Package, file_name: str, encoding: str = "utf-8"
) -> Iterable[AnyStr]:
    with get_file(package, file_name) as file:
        yield from map(lambda line: line.rstrip("\n"), file)


def read_bytes(package: Package, file_name: str) -> Iterable[bytes]:
    with get_file(package, file_name) as file:
        yield from file


def copy_file(package: Package, file_name: str, destination: PathType) -> None:
    data = read_bytes(package, file_name)
    with open(destination, "wb") as out_file:
        out_file.write(data)


def get_full_path(package: Package, file_name: str) -> str:
    resource = get_file(package, file_name)
    with resources.as_file(resource) as file:
        return file.as_posix()


def read_text_from_path(path: PathType, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as inp_file:
        return inp_file.read()


def save_text_to_path(text: str, path: PathType, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as out_file:
        out_file.write(text)
