import os
import subprocess


class TestBenchRunner:
    def __init__(self, workdir, files, test_bench_name):
        self._root = workdir
        self._ghdl_dir = "ghdl_build"
        self._files = files
        self._test_bench_name = test_bench_name

    @property
    def _workdir_flag(self):
        return f"--workdir={self._ghdl_dir}"

    def initialize(self):
        os.makedirs(f"{self._root}/{self._ghdl_dir}", exist_ok=True)
        subprocess.run(["ghdl", "-i", self._workdir_flag] + self._files, cwd=self._root)
        subprocess.run(
            [
                "ghdl",
                "-m",
                self._workdir_flag,
                "-fsynopsys",
                f"{self._test_bench_name}",
            ],
            cwd=self._root,
        )

    def run(self):
        subprocess.run(
            [
                "ghdl",
                "-r",
                self._workdir_flag,
                "-fsynopsys",
                f"{self._test_bench_name}",
            ],
            cwd=self._root,
        )
