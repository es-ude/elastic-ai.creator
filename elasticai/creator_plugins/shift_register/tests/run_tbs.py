from pathlib import Path

from elasticai.creator.ir2vhdl.testing import run_vunit_vhdl_testbenches

if __name__ == "__main__":
    run_vunit_vhdl_testbenches(
        [
            "elasticai.creator_plugins.shift_register",
            "elasticai.creator_plugins.counter",
        ],
        Path(__file__).parent,
    )
