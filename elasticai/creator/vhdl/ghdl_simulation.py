import glob
import os
import subprocess

from ._ghdl_report_parsing import parse_report


class GHDLSimulator:
    """Run a simulation tool for a given `top_design` and save whatever is written to stdout
    for subsequent inspection.

    This runner uses the GHDL tool.
    The parsed content has the following keys: `("source", "line", "column", "time", "type", "content")'
    """

    def __init__(self, workdir, top_design_name):
        self._root = workdir
        self._ghdl_dir = "ghdl_build"
        self._files = list(glob.glob(f"**/*.vhd", root_dir=self._root, recursive=True))
        self._standard = "08"
        self._result = {}
        self._test_bench_name = top_design_name

    def initialize(self):
        """Call this function once before calling `run()` and on every file change."""
        os.makedirs(f"{self._root}/{self._ghdl_dir}", exist_ok=True)
        self._load_files()
        self._compile()

    def run(self):
        """Runs the simulation and saves whatever the tool wrote to stdout.
        You're supposed to call `initialize` once, before calling `run`."""
        self._result = self._execute_command_and_return_stdout(
            self._assemble_command("-r") + [self._test_bench_name]
        )

    def getReportedContent(self) -> list[str]:
        """Strips any information that the simulation tool added automatically to the output
        to return only the information that was printed to stdout via VHDL/Verilog statements.
        """
        parsed = parse_report(self._result)
        return list(line["content"] for line in parsed)

    def getFullReport(self) -> list[dict]:
        """Parses the output from the simulation tool, to provide a more structured representation.
        The exact content depends on the simulation tool.
        """
        return parse_report(self._result)

    def getRawResult(self) -> str:
        """Returns the raw stdout output as written by the simulation tool."""
        return self._result

    def _load_files(self):
        self._execute_command(self._assemble_command("-i") + self._files)

    def _compile(self):
        self._execute_command(
            self._assemble_command("-m") + ["-fsynopsys", self._test_bench_name]
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
