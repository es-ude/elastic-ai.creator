from elasticai.creator.resource_utils import read_text, copy_file, PathType
from elasticai.creator.vhdl.resources import static_files


def read_static_file(file_name: str, encoding: str = "utf-8") -> str:
    return read_text(static_files, file_name, encoding)


def copy_static_file(file_name: str, destination: PathType) -> None:
    copy_file(static_files, file_name, destination)
