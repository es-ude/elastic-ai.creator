from collections.abc import Iterable, Iterator

import pytest

from elasticai.creator.hw_function_id import HwFunctionIdUpdater

pytestmark = pytest.mark.slow


def test_compute_id_ignores_target_file_contents(tmp_path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    source_file = build_dir / "source.vhd"
    source_file.write_text("signal a : std_logic;\n")
    target_file = build_dir / "skeleton_pkg.vhd"
    target_file.write_text(
        'constant SKELETON_ID : skeleton_id_t := (others => x"00");\n'
    )

    def passthrough(code: Iterable[str], _id: bytes) -> Iterator[str]:
        return iter(code)

    updater = HwFunctionIdUpdater(build_dir, target_file, passthrough)
    updater.compute_id()
    first_id = updater.id

    target_file.write_text(
        'constant SKELETON_ID : skeleton_id_t := (others => x"FF");\n'
    )
    updater.compute_id()
    second_id = updater.id

    assert first_id == second_id


def test_write_id_uses_replace_function_with_computed_id(tmp_path) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    source_file = build_dir / "source.vhd"
    source_file.write_text("entity test is\nend entity;\n")
    target_file = build_dir / "skeleton_pkg.vhd"
    target_file.write_text("placeholder\n")

    def replace_with_hex_id(_code: Iterable[str], hw_id: bytes) -> Iterator[str]:
        yield f"id={hw_id.hex()}"

    updater = HwFunctionIdUpdater(build_dir, target_file, replace_with_hex_id)
    updater.compute_id()
    expected = updater.id.hex()
    updater.write_id()

    assert target_file.read_text().strip() == f"id={expected}"
