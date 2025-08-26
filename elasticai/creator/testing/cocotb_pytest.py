import inspect
import json
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from functools import wraps
from os import environ
from pathlib import Path
from typing import Any

from elasticai.creator.file_generation.resource_utils import get_file_from_package

from .cocotb_prepare import (
    build_report_folder_and_testdata,
    read_testdata,
)
from .cocotb_runner import run_cocotb_sim


def _get_args(fn: Callable, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return inspect.signature(fn).bind_partial(*args, **kwargs).arguments


def create_name_for_build_test_subdir(
    top_module_name: str, fn: Callable, *args: Any, **kwargs: Any
) -> str:
    argstring = []
    for arg, value in _get_args(fn, *args, **kwargs).items():
        argstring.extend([str(arg), str(value)])
    if len(argstring) > 0:
        argstring = "_" + "_".join(argstring)
    else:
        argstring = ""
    return f"{top_module_name}_{fn.__name__}{argstring}"


def eai_testbench(fn):
    """
    Intended usage:

    ```python
    @cocotb.test
    @eai_testbench
    async def my_testbench_for_input_buffer(dut, cocotb_test_fixture, x):
      additional_input_data = cocotb_test_fixture.read()["input_data"]
      dut.d_in = x
    ```

    and

    ```python
    @pytest.mark.parametrize("x", [1, 2, 3])
    def test_input_buffer(cocotb_test_fixture, x):
      cocotb_test_fixture.write({"input_data": "hello world"})
      cocotb_test_fixture.run()
    ```

    The example will assume your toplevel module is `"input_buffer"` and
    it's source file lives in a sibling folder of the `test` folder that
    contains the pytest test function.
    It will create a unique subdirectory under `build_test` that matches
    the path to the module containing the testbench definition and pytest
    test function (those need to be the same).
    This prevents test A overriding the artifacts of test B.
    The name of the subdirectory will be derived from the parameters
    passed via the `parametrize` pytest marker and the top module name.
    In this example this results in folders: `input_buffer_x_1`, `input_buffer_x_2`, `input_buffer_x_3`.
    """
    build_test_dir = environ["EAI_SIM_TEST_DIR"]

    @wraps(fn)
    async def wrapper(dut):
        with open(build_test_dir / "testdata.json", "r") as f:
            kwargs = json.load(f)
        fn(dut=dut, **kwargs)

    return wrapper


class CocotbTestFixture:
    def __init__(
        self,
        test_fn: Callable,
        *args: float | int | str,
        **kwargs: float | int | str,
    ) -> None:
        self._test_fn = test_fn
        self._args = args
        self._kwargs = kwargs
        self._build_test_subdir = "<none>"
        self._top_module_name = self._test_fn.__name__.removeprefix("test_")
        self._id = "<none>"
        self._srcs = [
            get_file_from_package(
                test_fn.__module__, f"/../vhdl/{self._top_module_name}.vhd"
            )
        ]

    def setup(self):
        self._create_build_dir()

    def _create_build_dir(self):
        build_test_subdir = create_name_for_build_test_subdir(
            self._top_module_name, self._test_fn, *self._args, **self._kwargs
        )
        build_test_subdir = (
            self._test_fn.__module__.replace(".", "/") + "/" + build_test_subdir
        )

        self._build_test_subdir = build_report_folder_and_testdata(
            build_test_subdir, _get_args(self._test_fn, *self._args, **self._kwargs)
        )

    def write(self, data: dict[str, Any]) -> None:
        with open(self._build_test_subdir / Path("testdata.json"), "rw") as f:
            testdata = json.load(f)
            testdata.update(data)
            json.dump(testdata)

    def set_top_module_name(self, top_module_name: str) -> None:
        self._top_module_name = top_module_name

    def set_srcs(self, srcs: Iterable[str | Path]):
        self._srcs = list(srcs)

    def add_srcs(self, *srcs: str | Path) -> None:
        self._srcs.extend(srcs)

    def run(self, params, defines):
        environ["EAI_SIM_TEST_DIR"] = self._build_test_subdir
        with ExitStack() as stack:
            accessible_srcs = []
            for src in self._srcs:
                accessible_srcs.append(stack.enter_context(open(src, "r")))

            run_cocotb_sim(
                src_files=accessible_srcs,
                top_module_name=self._top_module_name,
                params=params,
                defines=defines,
            )


def cococtb_test_fixture(request):
    return CocotbTestFixture(request.function, **request.node.callspec.params)
