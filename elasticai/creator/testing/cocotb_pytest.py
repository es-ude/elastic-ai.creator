import hashlib
import importlib.resources
import inspect
import json
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from functools import wraps
from os import environ
from pathlib import Path
from typing import Any

import pytest

from .cocotb_prepare import build_report_folder_and_testdata
from .cocotb_runner import run_cocotb_sim


def eai_testbench(fn):
    """
    Intended usage:

    ```python
    @cocotb.test()
    @eai_testbench
    async def my_testbench_for_input_buffer(dut, x, input_data):
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
    test function (both need to live in the same module).
    This prevents test A overriding the artifacts of test B.
    The name of the subdirectory will be derived from the parameters
    passed via the `parametrize` pytest marker and the top module name.
    In this example this results in folders: `input_buffer_test_input_buffer_x_1`, `input_buffer_test_input_buffer_x_2`, `input_buffer_test_input_buffer_x_3`.

    """

    @wraps(fn)
    async def wrapper(dut):
        build_test_dir = environ["EAI_SIM_TEST_DIR"]
        with open(Path(build_test_dir) / "testdata.json", "r") as f:
            kwargs = json.load(f)
        await fn(dut=dut, **kwargs)

    return wrapper


class CocotbTestFixture:
    """Run cocotb via pytest, inject parameters to be available before and during test execution.

    The fixture will inspect the requesting test function to assume some default values and perform a little bit of setup. Namely this is

    * Use the test function name to determine the dut top module name and the name of its containing source file.
      These can be overriden inside the test function using `.set_top_module_name()` and `.set_srcs()`.
      The default name will be derived by stripping the `test_` prefix from the test function name.
      The implementation will try to find a vhdl or verilog file under `../{vhdl, verilog}/<name>.{vhd, v}`.
      Vhdl will take precedence. If no file is found, the initial srcs list will be left empty without raising an exception.

    * It will create a folder to contain test artifacts including waveforms, xml result, testdata json and compiled simulation
      object files. To avoid collisions, the name of the folder will be derived from the fully qualified test function name
      (replacing `.` by `/`) and the parameter list provided via pytest parametrization.

        The fixture assumes the test resolves package resources itself (e.g. via `get_file_from_package`).
        It will keep open the files whose paths are passed to `set_srcs()`/`add_srcs()` while `run` executes,
        but you must keep a surrounding `with` in the test so the package helper holds the resource alive during the simulation.

    * If you need to generate hdl sources prior to running testbenches and want to store them with the rest of the testing
      artifacts, you can retrieve the automatically determined folder via the `.get_artifact_dir()` method.
      This allows you to store the sources there and pass the resulting paths to the fixture using eg. `.add_srcs`.

    This is not intended to be used directly. Request `cocotb_test_fixture` as a pytest fixture instead.
    """

    def __init__(
        self,
        test_fn: Callable,
        *args: float | int | str,
        **kwargs: float | int | str,
    ) -> None:
        self._args = args
        self._kwargs = kwargs
        self._artifact_dir: Path | str = "<none>"
        if not (hasattr(test_fn, "__name__") and isinstance(test_fn.__name__, str)):
            raise TypeError("test_fn has to be a function-like object")
        self._test_fn = test_fn
        self._top_module_name = self._test_fn.__name__.removeprefix("test_")
        self._id = "<none>"
        self._timescale = ("1ps", "1fs")

        def get_parent_module(module: str):
            return ".".join(module.split(".")[:-1])

        self._context_stack = ExitStack()
        self._srcs: list[str] = []

    def setup(self):
        self._create_build_dir()
        self._set_default_srcs()

    def teardown(self):
        self._srcs = []
        self._context_stack.close()

    def _set_default_srcs(self) -> None:
        try:
            src_file = importlib.resources.path(
                ".".join(self._test_fn.__module__.split(".")[:-2] + ["vhdl"]),
                f"{self._top_module_name}.vhd",
            )
            self._srcs = [str(self._context_stack.enter_context(src_file))]
        except (ModuleNotFoundError, FileNotFoundError):
            try:
                src_file = importlib.resources.path(
                    ".".join(self._test_fn.__module__.split(".")[:-2] + ["verilog"]),
                    f"{self._top_module_name}.v",
                )
                self._srcs = [str(self._context_stack.enter_context(src_file))]
            except (ModuleNotFoundError, FileNotFoundError):
                pass

    def _create_build_dir(self):
        build_test_subdir = create_name_for_build_test_subdir(
            self._test_fn, *self._args, **self._kwargs
        )
        build_test_subdir = (
            self._test_fn.__module__.replace(".", "/") + "/" + build_test_subdir
        )

        self._artifact_dir = build_report_folder_and_testdata(
            build_test_subdir, _get_args(self._test_fn, *self._args, **self._kwargs)
        )

    def get_artifact_dir(self) -> Path:
        if self._artifact_dir == "<none>":
            raise ValueError("no artifact folder initialized")
        return Path(self._artifact_dir)

    def write(self, data: dict[str, Any]) -> None:
        with open(Path(self._artifact_dir) / Path("testdata.json"), "r") as f:
            testdata = json.load(f)
        testdata.update(data)
        with open(Path(self._artifact_dir) / Path("testdata.json"), "w") as f:
            json.dump(testdata, f)

    def set_top_module_name(self, top_module_name: str) -> None:
        self._top_module_name = top_module_name

    def set_srcs(self, srcs: Iterable[str | Path]):
        self._srcs = list((str(s) for s in srcs))

    def add_srcs(self, *srcs: str | Path) -> None:
        self._srcs.extend((str(s) for s in srcs))

    def set_timescale(self, scale: tuple[str, str]) -> None:
        self._timescale = scale

    def run(self, params, defines):
        environ["EAI_SIM_TEST_DIR"] = str(self._artifact_dir)

        run_cocotb_sim(
            src_files=self._srcs,
            top_module_name=self._top_module_name,
            cocotb_test_module=self._test_fn.__module__,
            params=params,
            defines=defines,
            build_sim_dir=self._artifact_dir,
            timescale=self._timescale,
        )


@pytest.fixture
def cocotb_test_fixture(request) -> Iterator[CocotbTestFixture]:
    """Yields the setup CocotbTestFixture and performs necessary clean up after the test run.

    To use the fixture either place add the line
    ```python
    pytest_plugins = "elasticai.creator.testing.cocotb_pytest"
    ```
    to either a conftest.py in the test directory tree or in the
    test module.

    For more information see the documentation of `CocotbTestFixture`
    """
    fixture = CocotbTestFixture(request.function, **request.node.callspec.params)
    fixture.setup()
    yield fixture
    fixture.teardown()


def _get_args(fn: Callable, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return inspect.signature(fn).bind_partial(*args, **kwargs).arguments


def create_name_for_build_test_subdir(fn: Callable, *args: Any, **kwargs: Any) -> str:
    mangling_args: list[str] = []

    def get_hash(v):
        # Deterministic hash with short output (16 hex chars = 64 bits)
        return hashlib.sha256(str(v).encode()).hexdigest()[:16]

    def to_string(v):
        if isinstance(v, (int, str)):
            return str(v)
        elif isinstance(v, float):
            return f"{v: .2f}"
        return str(get_hash(v))

    def make_hashable(value):
        if isinstance(value, Hashable):
            return value
        else:
            if isinstance(value, Sequence):
                return tuple((make_hashable(v) for v in value))
            elif isinstance(value, Mapping):
                return tuple(
                    (make_hashable(k), make_hashable(v)) for k, v in value.items()
                )

    for arg, value in _get_args(fn, *args, **kwargs).items():
        value = make_hashable(value)
        value = to_string(value)
        mangling_args.extend([str(arg), str(value)])
    if len(mangling_args) > 0:
        argstring = "_" + "_".join(mangling_args)
    else:
        argstring = ""
    fn_name = _get_name_from_function(fn)
    return f"{fn_name}{argstring}"


def _get_name_from_function(fn: Callable) -> str:
    if not (hasattr(fn, "__name__") and isinstance(fn.__name__, str)):
        raise TypeError("test_fn has to be a function-like object")
    return fn.__name__
