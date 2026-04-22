from pathlib import Path

import pytest

from .test_util import build_verilog_design as _build_verilog_design


@pytest.fixture
def build_verilog_design(tmpdir):
    def _wrapped(type: str, id: str, params: dict, build_tb: bool):
        return _build_verilog_design(
            type, id, params, ["multipliers"], build_tb, tmpdir
        )

    return _wrapped


@pytest.mark.slow
class TestBuildVerilogAdder:
    def test_build_mult_lut_signed_general_wo_tb(
        self, build_verilog_design, tmpdir: Path
    ):
        build_verilog_design(
            type="mult_lut_signed",
            id="0",
            params={"BITWIDTH": 3},
            build_tb=False,
        )
        assert (tmpdir / "mult_lut_signed_0.v").exists()
        assert (tmpdir / "adder_full.v").exists()
        assert (tmpdir / "adder_half.v").exists()

    def test_build_mult_dadda_signed_general_wo_tb(
        self, build_verilog_design, tmpdir: Path
    ):
        build_verilog_design(
            type="mult_lut_signed",
            id="2",
            params={"BITWIDTH": 4},
            build_tb=False,
        )
        assert (tmpdir / "mult_lut_signed_2.v").exists()
        assert (tmpdir / "adder_full.v").exists()
        assert (tmpdir / "adder_half.v").exists()

    def test_build_mult_lut_unsigned_general_wo_tb(
        self, build_verilog_design, tmpdir: Path
    ):
        build_verilog_design(
            type="mult_lut_unsigned",
            id="0",
            params={"BITWIDTH": 4},
            build_tb=False,
        )
        assert (tmpdir / "mult_lut_unsigned_0.v").exists()
        assert (tmpdir / "adder_full.v").exists()
        assert (tmpdir / "adder_half.v").exists()

    def test_build_mult_dadda_unsigned_general_wo_tb(
        self, build_verilog_design, tmpdir: Path
    ):
        build_verilog_design(
            type="mult_lut_unsigned",
            id="2",
            params={"BITWIDTH": 6},
            build_tb=False,
        )
        assert (tmpdir / "mult_lut_unsigned_2.v").exists()
        assert (tmpdir / "adder_full.v").exists()
        assert (tmpdir / "adder_half.v").exists()
