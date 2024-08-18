import os
import tempfile
import unittest

from elasticai.creator.nn.integer.vhdl_test_automation.create_makefile import (
    create_makefile,
)


class TestCreateMakefile(unittest.TestCase):
    def test_makefile_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_time = "5000ns"
            create_makefile(temp_dir, stop_time=stop_time)

            makefile_path = os.path.join(temp_dir, "makefile")
            self.assertTrue(os.path.exists(makefile_path))

            with open(makefile_path, "r") as file:
                content = file.read()
                self.assertIn(f"STOP_TIME = {stop_time}", content)
                self.assertIn("FILES = ./source/*/*.vhd", content)
                self.assertIn(
                    "GHDL_FLAGS  = --ieee=synopsys --warn-no-vital-generic", content
                )
                self.assertIn("WAVEFORM_VIEWER = gtkwave", content)

    def test_default_stop_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_makefile(temp_dir)

            makefile_path = os.path.join(temp_dir, "makefile")
            self.assertTrue(os.path.exists(makefile_path))

            with open(makefile_path, "r") as file:
                content = file.read()
                self.assertIn("STOP_TIME = 4000ns", content)
                self.assertIn("FILES = ./source/*/*.vhd", content)
