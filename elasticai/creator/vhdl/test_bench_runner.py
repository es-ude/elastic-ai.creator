import os
import subprocess
from typing import Union

from ._ghdl_report_parsing import parse


class TestBenchRunner:
    def __init__(self, workdir, files, test_bench_name):
        self._root = workdir
        self._ghdl_dir = "ghdl_build"
        self._files = files
        self._standard = "08"
        self._test_bench_name = test_bench_name
        self._result = {}

    def initialize(self):
        os.makedirs(f"{self._root}/{self._ghdl_dir}", exist_ok=True)
        self._load_files()
        self._compile()

    def run(self):
        self._run()

    def getReportedContent(self) -> list[str]:
        parsed = parse(self._result)
        return [line["content"] for line in parsed]

    def getFullReport(self) -> list[dict]:
        return parse(self._result)

    def getRawResult(self) -> str:
        return self._result

    def _load_files(self):
        self._execute_command(self._assemble_command("-i") + self._files)

    def _compile(self):
        self._execute_command(
            self._assemble_command("-m") + ["-fsynopsys", self._test_bench_name]
        )

    def _run(self):
        self._result = self._execute_command_and_return_stdout(
            self._assemble_command("-r") + [self._test_bench_name]
        )

    def _execute_command(self, command):
        return subprocess.run(command, cwd=self._root)

    def _execute_command_and_return_stdout(self, command):
        return subprocess.run(
            command, cwd=self._root, capture_output=True
        ).stdout.decode()

    def _assemble_command(self, command_flag):
        return ["ghdl", command_flag, f"--std={self._standard}", self._workdir_flag]

    @property
    def _workdir_flag(self):
        return f"--workdir={self._ghdl_dir}"
