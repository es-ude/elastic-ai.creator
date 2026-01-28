from collections.abc import Iterator
from importlib import resources
from pathlib import Path, PurePath
from typing import ContextManager

PathType = str | PurePath
Package = resources.Package


def get_file_from_package(package: Package, file_name: str) -> ContextManager[Path]:
    """
    This is a context manager, because the returned file might be extracted from a zip file and the context manager
    will take care of removing the resulting temporary files on __exit__
    """
    file_path = list(Path(file_name).parts)
    current_resource = resources.files(package)
    dir_contents = current_resource.iterdir()
    try:
        while True:
            resource = next(dir_contents)
            if resource.name == file_path[0]:
                if len(file_path) == 1:
                    return resources.as_file(resource)
                else:
                    file_path = file_path[1:]
                    dir_contents = resource.iterdir()
    except StopIteration:
        raise FileNotFoundError(
            f"The file '{file_name}' in package '{package}' does not exist."
        )


def read_text(package: Package, file_name: str) -> Iterator[str]:
    with get_file_from_package(package, file_name) as file:
        with open(file, "r") as opened_file:
            yield from map(lambda line: line.rstrip("\n"), opened_file)


def _read_bytes(package: Package, file_name: str) -> Iterator[bytes]:
    with get_file_from_package(package, file_name) as file:
        with open(file, "rb") as opened_file:
            yield from opened_file


def copy_file(package: Package, file_name: str, destination: PathType) -> None:
    with open(destination, "wb") as out_file:
        for data in _read_bytes(package=package, file_name=file_name):
            out_file.write(data)


def get_full_path(package: Package, file_name: str) -> str:
    with get_file_from_package(package, file_name) as file:
        return file.as_posix()


def read_text_from_path(path: PathType, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as inp_file:
        return inp_file.read()


def save_text_to_path(text: str, path: PathType, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as out_file:
        out_file.write(text)


def find_project_root(max_levels: int = 10) -> Path:
    cwd = Path(".").absolute()
    current = cwd

    def is_project_root(p):
        return (p / "pyproject.toml").exists()

    for _ in range(max_levels):
        if is_project_root(current):
            return current
        current = current.parent

    if is_project_root(current):
        return current
    return cwd
