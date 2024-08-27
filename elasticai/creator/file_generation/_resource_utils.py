from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import ContextManager

Package = resources.Package


def read_text(package: Package, file_name: str) -> Iterator[str]:
    with _get_file_from_package(package, file_name) as file:
        with open(file, "r") as opened_file:
            yield from map(lambda line: line.rstrip("\n"), opened_file)


def _get_file_from_package(package: Package, file_name: str) -> ContextManager[Path]:
    """
    This is a context manager, because the returned file might be extracted from a zip file and the context manager
    will take care of removing the resulting temporary files on __exit__
    """
    for resource in resources.files(package).iterdir():
        if resource.name == file_name:
            return resources.as_file(resource)
    raise FileNotFoundError(
        f"The file '{file_name}' in package '{package}' does not exist."
    )
