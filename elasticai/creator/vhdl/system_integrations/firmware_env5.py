from typing import Protocol

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.system_integrations.env5_constraints.env5_constraints import (
    ENV5Constraints,
)
from elasticai.creator.vhdl.system_integrations.middleware.middleware import Middleware
from elasticai.creator.vhdl.system_integrations.skeleton.skeleton import (
    LSTMSkeleton,
    Skeleton,
)
from elasticai.creator.vhdl.system_integrations.top.env5_reconfig_top import (
    ENV5ReconfigTop,
)


class SkeletonType(Protocol):
    def save_to(self, destination: Path) -> None: ...


class _FirmwareENv5Base:
    def __init__(self, skeleton: SkeletonType) -> None:
        self._skeleton = skeleton

    def save_to(self, destination: Path) -> None:
        def save_srcs(destination: Path):
            self._skeleton.save_to(destination.create_subpath("skeleton"))

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


class FirmwareENv5(_FirmwareENv5Base):
    def __init__(
        self,
        network: Design,
        x_num_values: int,
        y_num_values: int,
        id: list[int] | int,
        skeleton_version: str = "v1",
    ) -> None:
        super().__init__(
            skeleton=Skeleton(
                network_name=network.name,
                port=network.port,
                x_num_values=x_num_values,
                y_num_values=y_num_values,
                id=id,
                skeleton_version=skeleton_version,
            )
        )


class LSTMFirmwareENv5(_FirmwareENv5Base):
    def __init__(self, network: Design):
        super().__init__(skeleton=LSTMSkeleton(network.name))
