import os
import tempfile

from elasticai.creator.nn.integer.vhdl_test_automation.create_makefile import (
    create_makefile,
)


def test_makefile_creation():
    with tempfile.TemporaryDirectory() as temp_dir:
        stop_time = "5000ns"
        create_makefile(temp_dir, stop_time=stop_time)

        makefile_path = os.path.join(temp_dir, "makefile")
        assert os.path.exists(makefile_path)

        with open(makefile_path, "r") as file:
            content = file.read()
            assert f"STOP_TIME = {stop_time}" in content
            assert "FILES = ./source/*/*.vhd" in content
            assert "GHDL_FLAGS  = --ieee=synopsys --warn-no-vital-generic" in content
            assert "WAVEFORM_VIEWER = gtkwave" in content


def test_default_stop_time():
    with tempfile.TemporaryDirectory() as temp_dir:
        create_makefile(temp_dir)

        makefile_path = os.path.join(temp_dir, "makefile")
        assert os.path.exists(makefile_path)

        with open(makefile_path, "r") as file:
            content = file.read()
            assert "STOP_TIME = 4000ns" in content
            assert "FILES = ./source/*/*.vhd" in content
