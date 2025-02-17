from pathlib import Path

from elasticai.creator.ir2vhdl.testing import run_vunit_vhdl_testbenches

deps = ["elasticai.creator_plugins.sliding_window"]

if __name__ == "__main__":
    run_vunit_vhdl_testbenches(deps, Path(__file__).parent)
