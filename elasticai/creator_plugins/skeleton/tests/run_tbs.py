from pathlib import Path

from elasticai.creator.ir2vhdl.testing import run_vunit_vhdl_testbenches

if __name__ == "__main__":
    run_vunit_vhdl_testbenches(
        [f"elasticai.creator_plugins.{name}" for name in ["skeleton"]],  # type: ignore
        Path(__file__).parent,
        exclude=["skeleton.vhd", "buffered_network_wrapper.vhd"],
    )
