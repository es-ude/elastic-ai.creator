from typing import cast

import pytest

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.fixed_point import BatchNormedLinear


@pytest.fixture
def batchnorm_linear_design() -> BatchNormedLinear:
    return BatchNormedLinear(
        total_bits=16,
        frac_bits=8,
        in_features=3,
        out_features=2,
    )


def save_design(design: BatchNormedLinear) -> dict[str, str]:
    destination = InMemoryPath("batchnorm_linear", parent=None)
    design.create_design("batchnorm_linear").save_to(destination)
    files = cast(list[InMemoryFile], list(destination.children.values()))
    return {file.name: "\n".join(file.text) for file in files}


def test_saved_design_contains_needed_files(
    batchnorm_linear_design: BatchNormedLinear,
) -> None:
    saved_files = save_design(batchnorm_linear_design)
    expected_files = {
        "batchnorm_linear_rom.vhd",
        "batchnorm_linear.vhd",
    }
    actual_files = set(saved_files.keys())
    assert expected_files == actual_files
