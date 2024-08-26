from pathlib import Path
from typing import Any

from elasticai.creator.file_generation.v2.savable import Savable
from elasticai.creator.file_generation.v2.temporary import TemporaryDirectory


def design_file_structure(design: Savable) -> dict[str, Any]:
    with TemporaryDirectory() as destination:
        design.save_to(destination)

        def load_files(directory: Path) -> dict[str, Any]:
            files = list(directory.glob("*"))
            structure = dict()
            for file in files:
                if file.is_dir():
                    structure[file.name] = load_files(file)
                else:
                    structure[file.name] = file.read_text()
            return structure

        return load_files(destination)
