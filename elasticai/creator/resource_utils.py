from importlib import resources
from importlib.abc import Traversable
from pathlib import PurePath
from typing import Union

PathType = Union[str, PurePath]


def _get_file(package: resources.Package, file_name: str) -> Traversable:
    for resource in resources.files(package).iterdir():
        if resource.name == file_name:
            return resource
    raise FileNotFoundError(f"The file '{file_name}' does not exist.")


def read_text(
    package: resources.Package, file_name: str, encoding: str = "utf-8"
) -> str:
    return _get_file(package, file_name).read_text(encoding)


def read_bytes(package: resources.Package, file_name: str) -> bytes:
    return _get_file(package, file_name).read_bytes()


def copy_file(
    package: resources.Package, file_name: str, destination: PathType
) -> None:
    data = read_bytes(package, file_name)
    with open(destination, "wb") as out_file:
        out_file.write(data)


def get_full_path(package: resources.Package, file_name: str) -> str:
    resource = _get_file(package, file_name)
    with resources.as_file(resource) as file:
        return file.as_posix()
