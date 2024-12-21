import logging as _logger
import importlib.resources as res
from typing import Iterator
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
import asyncio as a
from asyncio.subprocess import Process
from typing import Coroutine, Any, TypeVar, Callable, AsyncIterable
import re
import sys
from ._run import run_and_wait, run_and_process_pipes

CREATOR_PLUGIN_NAMESPACE = "elasticai.creator_plugins"


def _get_logger() -> _logger.Logger:
    return _logger.getLogger(__name__)


def get_paths_from_package(package: str) -> Iterator[Path]:
    for item in res.files(package).iterdir():
        with res.as_file(item) as f:
            yield f


def check_for_ghdl():
    ghdl_path = run_and_wait("which", "ghdl").stdout
    log = _get_logger()
    if ghdl_path == [] or ghdl_path == [""]:
        log.error("ghdl not found in $PATH")
    exit(1)


def ghdl_command(command: str, *args: str) -> tuple[str, ...]:
    return "ghdl", command, "--std=08", "-fsynopsys", *args


def ghdl_import(*files: Path) -> None:
    run_and_wait(*ghdl_command("-i", *tuple(str(f.absolute()) for f in files)))


class TestBenchReport:
    def __init__(self, root: ET.Element):
        self.root = root
        self.failures = 0
        self.successes = 0

    def new_test(self, name: str, file: str):
        self.current_test = ET.SubElement(
            self.root, "testsuite", attrib=dict(name=name, file=file)
        )

    def record_stdout(self, text: str):
        stdout = ET.SubElement(self.current_test, "system-out")
        stdout.text = text

    def record_stderr(self, text: str):
        stderr = ET.SubElement(self.current_test, "system-err")
        stderr.text = text

    def record_failure(self):
        self.failures += 1

    def record_success(self):
        self.successes += 1

    def finish_test(self):
        self.current_test.attrib.update(
            {
                "tests": str(self.failures + self.successes),
                "failures": str(self.failures),
                "successes": str(self.successes),
            }
        )
        self.failures = 0
        self.successes = 0

    def dump(self, to: str):
        with open(to, "wb") as f:
            f.write(ET.tostring(self.root))


class TestBench:
    def __init__(self, file: str, name: str):
        self.file = file
        self.name = name

    def __repr__(self) -> str:
        file, name = self.file, self.name
        return f"TestBench({file=}, {name=})"


class VhdlTestBenchRunner:
    def __init__(self, report: TestBenchReport) -> None:
        self.vhd_files: list[Path] = []
        self._log = _get_logger()
        self._report = report
        self._tb_regex = re.compile(r".*?(\w+_tb)\.vhd")

    def run(self) -> None:
        self._collect_files()
        self._import_files()
        for tb in self._get_testbenches():
            self._log.debug("starting tb {}".format(tb))
            self._new_test(tb)
            success = self._compile_tb(tb)
            if success:
                success = self._run_tb(tb)
            if success:
                self._report.record_success()
            self._report.finish_test()

    def _new_test(self, tb: TestBench) -> None:
        self._report.new_test(tb.name, tb.file)

    def _get_testbenches(self) -> list[TestBench]:
        tbs = []
        for f in self.vhd_files:
            absolute = str(f.absolute())
            m = self._tb_regex.match(absolute)
            if m:
                tbs.append(TestBench(absolute, m.groups()[-1]))
        return tbs

    def _compile_tb(self, tb: TestBench) -> bool:
        return_code = self._run_and_record_ghdl("-m", tb.name)
        if return_code != 0:
            self._report.record_failure()
            return False
        return True

    def _run_tb(self, tb: TestBench) -> bool:
        return_code = self._run_and_record_ghdl("-r", tb.name)
        if return_code != 0:
            self._report.record_failure()
            return False
        return True

    def _run_and_record_ghdl(self, command: str, *args: str) -> bool:
        command, *_args = ghdl_command(command, *args)
        result = run_and_process_pipes(
            command, self._collect_stdout, self._collect_stderr, *_args
        )
        return result.returncode

    def _import_files(self) -> None:
        ghdl_import(*self.vhd_files)

    async def _collect_stdout(self, stream: AsyncIterable[bytes]) -> None:
        lines: list[str] = []
        async for line in stream:
            lines.append(line.decode())
        self._report.record_stdout("".join(lines))

    async def _collect_stderr(self, stream: AsyncIterable[bytes]) -> None:
        lines: list[str] = []
        async for line in stream:
            text = line.decode()
            if text.strip() != "":
                sys.stderr.write("\x1b[31;1mghdl error:\x1b[0m {}".format(text))
                lines.append(text)
        self._report.record_stderr("".join(lines))

    def _collect_files(self) -> None:
        self._log.debug("collecting vhd files")
        for f in get_paths_from_package(CREATOR_PLUGIN_NAMESPACE):
            for sub_path in f.glob("**/*.vhd"):
                if "middleware" not in sub_path.parts:
                    self._log.debug("collecting {}".format(sub_path.name))
                    self.vhd_files.append(sub_path)
