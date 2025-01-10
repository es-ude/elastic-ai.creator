import asyncio as a
import importlib.resources as res
import logging
import logging as _logger
import re
import sys
import xml.etree.ElementTree as ET
from asyncio.subprocess import Process
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterable, Callable, Coroutine, Iterator, Literal, TypeVar

from ._console_out import Printer
from ._run import run_and_process_pipes, run_and_wait

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
    def __init__(
        self,
    ):
        self.root = ET.Element("testsuites", attrib=dict(tests="0", failures="0"))

    def new_suite(self, name: str, file: str):
        self.current_suite = ET.SubElement(
            self.root,
            "testsuite",
            attrib=dict(name=name, file=file, tests="0", failures="0"),
        )

    def num_failures(self) -> int:
        return int(self.root.get("failures", "0"))

    def num_tests(self) -> int:
        return int(self.root.get("tests", "0"))

    def new_case(self, name: str, file: str):
        self.current_test = ET.SubElement(
            self.current_suite, "testcase", attrib=dict(name=name)
        )
        self._increment_tests(self.current_suite)
        self._increment_tests(self.root)

    def record_failure(self, msg: str, type: Literal["warning", "error"]) -> None:
        self._increment_failures(self.current_suite)
        self.current_test.append(
            ET.Element("failure", attrib=dict(message=msg, type=type))
        )
        self._increment_failures(self.root)

    def dump(self, file: BytesIO):
        file.write(ET.tostring(self.root))

    @staticmethod
    def _increment_failures(element: ET.Element) -> None:
        TestBenchReport._increment_field(element, "failures")

    @staticmethod
    def _increment_tests(element: ET.Element) -> None:
        TestBenchReport._increment_field(element, "tests")

    @staticmethod
    def _increment_field(element: ET.Element, field: str) -> None:
        value = element.get(field, "0")
        value = str(int(value) + 1)
        element.set(field, value)

    def _print_errors_for_failed_tests(self, p: Printer) -> Iterable[str]:
        printed = False
        for suite in self.root.iter("testsuite"):
            failed = suite.get("failures", "0")
            if failed != "0":
                printed = True
                yield f"===== {suite.get('name')} errors ========"
                for case in suite.iter("testcase"):
                    for failure in case.iter("failure"):
                        yield p.red(failure.get("message"))
        if printed:
            yield "=" * 60
            yield ""

    def _print_each_test_suite(self, p: Printer) -> Iterable[str]:
        checkmark = p.green_bold(b"\xe2\x9c\x93".decode("utf-8"))
        crossmark = p.red_bold(b"\xcb\x9f".decode("utf-8"))
        result_column = 50
        move_cursor_to_result_col = f"\x1b[{result_column}G"
        for suite in self.root.iter("testsuite"):
            failed = suite.get("failures", "0")
            if failed != "0":
                yield f"{p.red_bold(suite.get('name'))}:{move_cursor_to_result_col} {p.red(failed)}{crossmark}"
            else:
                yield f"{suite.get('name')}:{move_cursor_to_result_col} {p.green(suite.get('tests'))}{checkmark}"

    def _print_totals(self, p: Printer) -> Iterable[str]:
        total_fails = self.root.get("failures")
        total_tests = self.root.get("tests")
        total_passed = str(int(total_tests) - int(total_fails))
        totals = []
        if total_passed != "0":
            totals.append(p.green(f"passed: {total_passed}"))
        if total_fails != "0":
            totals.append(p.red(f"failed: {total_fails}"))

        total = ", ".join(totals)
        yield ""
        yield f"{'=' * 20} {total} {'=' * 20}"

    def pretty_print(self) -> Iterable[str]:
        p = Printer()
        yield ""
        yield from self._print_errors_for_failed_tests(p)
        yield from self._print_each_test_suite(p)
        yield from self._print_totals(p)


class TestBench:
    def __init__(self, file: str, name: str):
        self.file = file
        self.name = name

    def __repr__(self) -> str:
        file, name = self.file, self.name
        return f"TestBench({file=}, {name=})"


@dataclass
class GhdlMsg:
    type: Literal["assertion error", "simulation finished"]
    file: str
    line: int
    column: int
    time: str
    msg: str

    def render(self) -> str:
        return (
            "ghdl output:\n"
            f"\ttype: {self.type}\n"
            f"\tfile: {self.file}:{self.line}:{self.column}\n"
            f"\ttime: {self.time}\n"
            f"\tmsg: {self.msg}"
        )


@dataclass
class _CondensedGhdlMsg:
    type: Literal["assertion error", "simulation finished"]
    file: str
    line: int
    column: int
    times: list[str]
    msg: str

    def render(self) -> str:
        return (
            "ghdl output:\n"
            f"\ttype: {self.type}\n"
            f"\tfile: {self.file}:{self.line}:{self.column}\n"
            f"\ttimes: {', '.join(self.times)}\n"
            f"\tmsg: {self.msg}"
        )

    @classmethod
    def from_msg(cls, m: GhdlMsg) -> "_CondensedGhdlMsg":
        return cls(m.type, m.file, m.line, m.column, [m.time], m.msg)

    def key(self) -> int:
        return hash((self.type, self.file, self.line, self.column, self.msg))

    def merge(self, other: "_CondensedGhdlMsg") -> None:
        self.times.extend(other.times)


def parse_ghdl_msg(txt: str) -> list[GhdlMsg]:
    msgs = []
    for line in txt.splitlines():
        parts = line.split(":", maxsplit=5)
        if len(parts) == 6:
            msg = GhdlMsg(
                type="assertion error",
                msg=parts[5].strip(),
                file=parts[0],
                line=int(parts[1]),
                column=int(parts[2]),
                time=parts[3][1:],
            )
            msgs.append(msg)
        elif line.startswith("simulation finished"):
            msgs.append(
                GhdlMsg(
                    type="simulation finished",
                    file=".",
                    line=-1,
                    column=-1,
                    time=line.strip("simulation finished @"),
                    msg="",
                )
            )
        elif len(msgs) > 1:
            msgs[-1].msg += line

    return msgs


class VhdlTestBenchRunner:
    def __init__(self, report: TestBenchReport) -> None:
        self.vhd_files: list[Path] = []
        self._log = _get_logger()
        self._report = report
        self._tb_regex = re.compile(r".*?(\w+_tb)\.vhd")
        self._last_stderr = ""
        self._last_stdout = ""

    def run(self) -> None:
        self._collect_files()
        self._import_files()
        for tb in self._get_testbenches():
            self._log.debug("starting tb {}".format(tb))
            self._new_test(tb)
            success = self._compile_tb(tb)
            if success:
                self._run_tb(tb)

    def _new_test(self, tb: TestBench) -> None:
        self._report.new_suite(tb.name, tb.file)
        self._report.new_case(tb.name, tb.file)

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
            self._report.record_failure(self._last_stderr, "error")
            return False
        return True

    def _run_tb(self, tb: TestBench) -> None:
        return_code = self._run_and_record_ghdl("-r", tb.name)
        if return_code != 0:
            self._report.record_failure(self._last_stderr, "error")
        else:
            msgs = parse_ghdl_msg(self._last_stdout)
            msgs = [m for m in msgs if m.type == "assertion error"]
            reduced_msgs: dict[int, _CondensedGhdlMsg] = {}
            for m in msgs:
                c = _CondensedGhdlMsg.from_msg(m)
                if c.key() in reduced_msgs:
                    reduced_msgs[c.key()].merge(c)
                else:
                    reduced_msgs[c.key()] = c

            rendered = "\n".join(msg.render() for msg in reduced_msgs.values())
            if len(rendered) > 0:
                self._report.record_failure(rendered, "error")

    def _run_and_record_ghdl(self, command: str, *args: str) -> bool:
        command, *_args = ghdl_command(command, *args)
        result = run_and_process_pipes(
            command, self._collect_stdout, self._collect_stderr, *_args
        )
        return result.returncode

    def _import_files(self) -> None:
        ghdl_import(*self.vhd_files)

    async def _collect_stdout(self, stream: AsyncIterable[bytes]) -> None:
        """
        If we'd want to detect individual test cases from ghdl output,
        we could parse that output here.
        """
        lines: list[str] = []
        async for line in stream:
            lines.append(line.decode())
            self._log.debug(lines[-1])
        self._last_stdout = "".join(lines)

    async def _collect_stderr(self, stream: AsyncIterable[bytes]) -> None:
        """
        If we'd want to detect individual test cases from ghdl output,
        we could parse that output here.
        """
        lines: list[str] = []
        async for line in stream:
            text = line.decode()
            if text.strip() != "":
                self._log.error("\x1b[31;1mghdl error:\x1b[0m {}".format(text))
                lines.append(text)
        self._last_stderr = "".join(lines)

    def _collect_files(self) -> None:
        self._log.debug("collecting vhd files")
        for f in get_paths_from_package(CREATOR_PLUGIN_NAMESPACE):
            for sub_path in f.glob("**/*.vhd"):
                if "middleware" not in sub_path.parts:
                    self._log.debug("collecting {}".format(sub_path.name))
                    self.vhd_files.append(sub_path)
