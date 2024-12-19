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

CREATOR_PLUGIN_NAMESPACE = "elasticai.creator_plugins"


def _get_logger() -> _logger.Logger:
    return _logger.getLogger(__name__)


def get_paths_from_package(package: str) -> Iterator[Path]:
    for item in res.files(package).iterdir():
        with res.as_file(item) as f:
            yield f


def run(program: str, *args: str) -> Coroutine[Any, Any, Process]:
    logger = _get_logger()
    logger.debug(
        "run {} {}".format(
            program,
            "...".join((" ".join(args[0:2]), " ".join(args[-2:])))
            if len(args) > 4
            else " ".join(args),
        )
    )

    return a.create_subprocess_exec(
        program, *args, stdout=a.subprocess.PIPE, stderr=a.subprocess.PIPE
    )


class FinishedProcess:
    def __init__(self, stdout: list[str], stderr: list[str], returncode: bool):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def run_and_process_pipes(
    program: str,
    stdout_fn: Callable[[AsyncIterable[bytes]], Coroutine[None, None, _T1]],
    stderr_fn: Callable[[AsyncIterable[bytes]], Coroutine[None, None, _T2]],
    *args: str,
) -> tuple[bool, _T1, _T2]:
    async def _run():
        process = await run(program, *args)
        async with a.TaskGroup() as g:
            stdout_t = g.create_task(stdout_fn(process.stdout))
            stderr_t = g.create_task(stderr_fn(process.stderr))
        return process.returncode, stdout_t.result(), stderr_t.result()

    return a.run(_run())


def run_and_wait(program, *args) -> FinishedProcess:
    async def run_and_save(
        stream: AsyncIterable[bytes],
    ) -> list[str]:
        lines = []
        async for line in stream:
            lines.append(line.decode())
        return lines

    return_code, stdout, stderr = run_and_process_pipes(
        program, run_and_save, run_and_save, *args
    )
    return FinishedProcess(stdout, stderr, return_code)


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

    def dump(self):
        with open("testbenches_report.xml", "wb") as f:
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
        return_code, *_ = run_and_process_pipes(
            command, self._collect_stdout, self._collect_stderr, *_args
        )
        return return_code

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = _get_logger()
    report = TestBenchReport(ET.Element("testsuites"))
    logger.debug("running testbenches")
    test_benches = VhdlTestBenchRunner(report)
    test_benches.run()
    report.dump()
