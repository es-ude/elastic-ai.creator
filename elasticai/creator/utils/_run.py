import logging as _logger
from typing import Any, Coroutine, TypeVar, Generic
from asyncio.subprocess import Process as _Process
import asyncio as a
from collections.abc import Callable, AsyncIterable


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


class FinishedProcess(Generic[_T1, _T2]):
    def __init__(self, stdout: _T1, stderr: _T2, returncode: bool):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def run_and_process_pipes(
    program: str,
    stdout_fn: Callable[[AsyncIterable[bytes]], Coroutine[None, None, _T1]],
    stderr_fn: Callable[[AsyncIterable[bytes]], Coroutine[None, None, _T2]],
    *args: str,
) -> FinishedProcess[_T1, _T2]:
    """Run `program` and use `*_fn` to process `stdout`/`stdin` respectively.

    `stdout_fn` and `stdin_fn` will be run asynchronously.
    """

    async def run():
        process = await _run(program, *args)
        async with a.TaskGroup() as g:
            stdout_t = g.create_task(stdout_fn(process.stdout))
            stderr_t = g.create_task(stderr_fn(process.stderr))
        return process.returncode, stdout_t.result(), stderr_t.result()

    return_code, stdout, stderr = a.run(run())
    return FinishedProcess(stdout, stderr, return_code)


def run_and_wait(program, *args) -> FinishedProcess:
    async def run_and_save(
        stream: AsyncIterable[bytes],
    ) -> list[str]:
        lines = []
        async for line in stream:
            lines.append(line.decode())
        return lines

    return run_and_process_pipes(program, run_and_save, run_and_save, *args)


def _get_logger() -> _logger.Logger:
    return _logger.getLogger(__name__)


def _run(program: str, *args: str) -> Coroutine[Any, Any, _Process]:
    """"""
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
