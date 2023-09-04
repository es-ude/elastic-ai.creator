import os
import subprocess


class TestBenchRunner:
    def __init__(self, workdir, files, test_bench_name):
        self._root = workdir
        self._ghdl_dir = "ghdl_build"
        self._files = files
        self._standard = "08"
        self._test_bench_name = test_bench_name

    def initialize(self):
        os.makedirs(f"{self._root}/{self._ghdl_dir}", exist_ok=True)
        self._load_files()
        self._compile()

    def run(self):
        self._run()

    def _load_files(self):
        self._execute_command(self._assemble_command("-i") + self._files)

    def _compile(self):
        self._execute_command(
            self._assemble_command("-m") + ["-fsynopsys", self._test_bench_name]
        )

    def _run(self):
        output = self._execute_command_and_return_stdout(
            self._assemble_command("-r") + [self._test_bench_name]
        )

    def _execute_command(self, command):
        return subprocess.run(command, cwd=self._root)

    def _execute_command_and_return_stdout(self, command):
        return subprocess.run(command, cwd=self._root, capture_output=True).stdout

    def _assemble_command(self, command_flag):
        return ["ghdl", command_flag, f"--std={self._standard}", self._workdir_flag]

    @property
    def _workdir_flag(self):
        return f"--workdir={self._ghdl_dir}"
