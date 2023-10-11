from elasticai.creator.file_generation.savable import Path
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design_creator import DesignCreator
from elasticai.creator.vhdl.system_integrations.env5_constraints.env5_constraints import (
    ENV5Constraints,
)
from elasticai.creator.vhdl.system_integrations.middleware.middleware import Middleware
from elasticai.creator.vhdl.system_integrations.skeleton.skeleton import Skeleton
from elasticai.creator.vhdl.system_integrations.top.env5_reconfig_top import (
    ENV5ReconfigTop,
)


class FirmwareENv5:
    def __init__(self, network: Design) -> None:
        self._network = network

    def save_to(self, destination: Path) -> None:
        def save_srcs(destination: Path):
            skeleton = Skeleton(self._network.name, self._network.port)
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
