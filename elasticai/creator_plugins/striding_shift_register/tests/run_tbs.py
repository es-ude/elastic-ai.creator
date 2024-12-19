from pathlib import Path

from elasticai.creator.ir2vhdl.testing import run_vunit_vhdl_testbenches

deps = [
    "elasticai.creator_plugins.{name}".format(name=name)
    for name in [
        "striding_shift_register",
        "shift_register",
        "counter",
    ]
]


if __name__ == "__main__":
    run_vunit_vhdl_testbenches(deps, Path(__file__).parent)
