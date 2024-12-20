import warnings
from typing import Protocol

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.system_integrations.env5_constraints.env5_constraints import (
    ENV5Constraints,
)
from elasticai.creator.vhdl.system_integrations.middleware.middleware import Middleware
from elasticai.creator.vhdl.system_integrations.skeleton.skeleton import (
    EchoSkeletonV2,
    LSTMSkeleton,
)
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
    def create_design(self, name: str) -> Design: ...

    def create_testbench(self, name: str, design: Design) -> Design: ...


class FirmwareLSTMENv5:
    def __init__(self, model: _DesignAndTestbenchCreator):
        self._network = model.create_design("network")
        self._testbench = model.create_testbench("network_tb", self._network)

    def save_to(self, destination: Path):
        def save_srcs(destination: Path):
            lstm_skeleton = LSTMSkeleton(self._network.name)
            lstm_skeleton.save_to(destination.create_subpath("skeleton"))

            self._network.save_to(destination.create_subpath(self._network.name))

            middleware = Middleware()
            middleware.save_to(destination.create_subpath("middleware"))
            self._testbench.save_to(
                destination.create_subpath("test_benches").create_subpath(
                    self._testbench.name
                )
            )
            env5_reconfig_top = ENV5ReconfigTop()
            env5_reconfig_top.save_to(destination.create_subpath("env5_reconfig_top"))

        def save_constraints(destination: Path):
            env5_config = ENV5Constraints()
            env5_config.save_to(destination.create_subpath("env5_config"))

        srcs = destination.create_subpath("srcs")
        constraints = destination.create_subpath("constraints")
        save_constraints(constraints)
        save_srcs(srcs)


class FirmwareEchoServerSkeletonV2:
    def __init__(self, num_inputs: int, bitwidth: int):
        self._num_inputs = num_inputs
        self._bitwidth = bitwidth
        self.skeleton_id: list[int] = list()

    def save_to(self, destination: Path):
        def save_srcs(destination: Path):
            skeleton = EchoSkeletonV2(self._num_inputs, bitwidth=self._bitwidth)
            self.skeleton_id = skeleton._id
            skeleton.save_to(destination.create_subpath("skeleton"))

            middleware = Middleware()
            middleware.save_to(destination.create_subpath("middleware"))

            env5_reconfig_top = ENV5ReconfigTop()
            env5_reconfig_top.save_to(destination.create_subpath("env5_reconfig_top"))

        def save_constraints(destination: Path):
            env5_config = ENV5Constraints()
            env5_config.save_to(destination.create_subpath("env5_config"))

        srcs = destination.create_subpath("srcs")
        constraints = destination.create_subpath("constraints")
        save_constraints(constraints)
        save_srcs(srcs)
