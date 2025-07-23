from importlib import resources
from logging import getLogger
from pathlib import Path

from vunit import VUnit


def run_vunit_vhdl_testbenches(
    deps: list[str], test_dir: Path | str, exclude: list[str] = None
):
    logger = getLogger(__name__)
    if exclude is None:
        exclude = []
    if isinstance(test_dir, str):
        test_dir = Path(test_dir)
    test_dir = test_dir.absolute()
    logger.info("using test_dir {}".format(test_dir))
    vu = VUnit.from_argv()
    vu.add_vhdl_builtins()
    lib = vu.add_library("lib")
    for testbench in test_dir.glob("*_tb.vhd"):
        logger.info("adding testbench {}".format(testbench))
        lib.add_source_file(testbench.absolute())

    def should_be_excluded(file_name):
        for expr in exclude:
            if file_name.endswith(expr):
                return True
        return False

    for dep in deps:
        vhdl_dir = resources.files(dep).joinpath("vhdl")
        if not vhdl_dir.is_dir():
            raise IOError("could not find directory under {}".format(vhdl_dir))
        for file in vhdl_dir.glob("*.vhd"):  # type: ignore
            file_name = str(file.absolute())
            if should_be_excluded(file_name):
                continue
            lib.add_source_files(file_name)
    vu.main()
