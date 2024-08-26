import warnings
from pathlib import Path
from typing import Protocol

from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.system_integrations.env5_constraints.env5_constraints import (
    ENV5Constraints,
)
from elasticai.creator.vhdl.system_integrations.middleware.middleware import Middleware
from elasticai.creator.vhdl.system_integrations.skeleton.skeleton import LSTMSkeleton
from elasticai.creator.vhdl.system_integrations.top.env5_reconfig_top import (
    ENV5ReconfigTop,
)

warnings.warn(
    message=(
        "Will be removed. Use LSTMFirmwareENV5 from the"
        " elasticai.creator.vhdl.system_integrations.firmware_env5 package"
        " instead."
    ),
    category=DeprecationWarning,
)


class _DesignAndTestbenchCreator(Protocol):
    def create_design(self, name: str) -> Design:
        ...

    def create_testbench(self, name: str, design: Design) -> Design:
        ...


class FirmwareENv5:
    def __init__(self, model: _DesignAndTestbenchCreator):
        self._network = model.create_design("network")
        self._testbench = model.create_testbench("network_tb", self._network)

    def save_to(self, destination: Path):
        def save_srcs(destination: Path):
            lstm_skeleton = LSTMSkeleton(self._network.name)
            lstm_skeleton.save_to(destination / "skeleton")

            self._network.save_to(destination / self._network.name)

            middleware = Middleware()
            middleware.save_to(destination / "middleware")
            self._testbench.save_to(destination / "test_benches" / self._testbench.name)
            env5_reconfig_top = ENV5ReconfigTop()
            env5_reconfig_top.save_to(destination / "env5_reconfig_top")

        def save_constraints(destination: Path):
            env5_config = ENV5Constraints()
            env5_config.save_to(destination / "env5_config")

        save_constraints(destination / "constraints")
        save_srcs(destination / "srcs")
