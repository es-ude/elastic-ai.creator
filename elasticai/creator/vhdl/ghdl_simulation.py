import glob
import os
import subprocess

from ._ghdl_report_parsing import parse_report

class SimulationError(Exception):
    pass


class GHDLSimulator:
    """Run a simulation tool for a given `top_design` and save whatever is written to stdout
    for subsequent inspection.

    This runner uses the GHDL tool.
    The parsed content has the following keys: `("source", "line", "column", "time", "type", "content")'

    Will raise a `SimulationError` in case any of the calls to ghdl in the steps `initialize` or `run` fails.
    Args:
        workdir: typically the path to your build root, this is where we will look for vhd files
    """

    def __init__(self, workdir, top_design_name) -> None:
        self._root = workdir
        self._ghdl_dir = "ghdl_build"
        self._files = list(glob.glob(f"**/*.vhd", root_dir=self._root, recursive=True))
        self._standard = "08"
        self._test_bench_name = top_design_name
        self._generics: dict[str, str] = {}
        self._error_message = ""
        self._completed_process: None | subprocess.CompletedProcess = None

    def add_generic(self, **kwargs):
        self._generics.update(kwargs)

    def initialize(self):
        """Call this function once before calling `run()` and on every file change."""
        os.makedirs(f"{self._root}/{self._ghdl_dir}", exist_ok=True)
        self._load_files()
        self._compile()

    def run(self):
        """Runs the simulation and saves whatever the tool wrote to stdout.
        You're supposed to call `initialize` once, before calling `run`."""
        generic_options = [f"-g{key}={value}" for key, value in self._generics.items()]
        self._execute_command(
            self._assemble_command(["-r"]) + ["-fsynopsys", self._test_bench_name] + generic_options
        )

    @property
    def _result(self) -> str:
        return self._stdout()

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
        self._completed_process = subprocess.run(command, cwd=self._root, capture_output=True)
        self._check_for_error()

    def _check_for_error(self):
        try:
            self._completed_process.check_returncode()
        except subprocess.CalledProcessError as exception:
            error_message = self._get_error_message()
            sim_error = SimulationError(error_message)
            raise sim_error from exception

    def _get_error_message(self) -> str:
        # ghdl seems to pipe errors to stdout instead of stdin
        error_message = self._stderr()
        if error_message == "":
            error_message = self._stdout()
        return error_message

    def _stdout(self) -> str:
        if self._completed_process is not None:
            return self._completed_process.stdout.decode()
        else:
            return ""

    def _stderr(self) -> str:
        if self._completed_process is not None:
            return self._completed_process.stderr.decode()
        else:
            return ""

    def _assemble_command(self, command_flags):
        if isinstance(command_flags, str):
            command_flags = [command_flags]
        return (
            ["ghdl"] + command_flags + [f"--std={self._standard}", self._workdir_flag]
        )

    @property
    def _workdir_flag(self):
        return f"--workdir={self._ghdl_dir}"
