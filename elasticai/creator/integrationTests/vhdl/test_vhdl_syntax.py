import subprocess
import unittest
import glob


class VHDLSyntaxTest(unittest.TestCase):
    def test_syntax_of_all_vhdl_files(self):
        # IMPORTANT: not all shells support the wildcard ** as replacement for all directories and subdirectories
        # Therefore, we first select all vhd files with the glob library and execute the syntax check for all of them
        vhd_files = glob.glob("../**/*.vhd", recursive=True)
        for vhd_file in vhd_files:
            with self.subTest(vhd_file):
                process = subprocess.Popen(
                    "ghdl -s " + vhd_file,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                stdout, stderr = process.communicate()
                self.assertEqual(
                    "", stderr, msg="Syntax check of vhdl file " + vhd_file + " failed."
                )
