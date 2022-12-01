from typing import Iterable

from elasticai.creator.vhdl.vhdl_files import (
    VHDLModule,
    VHDLBaseModule,
    VHDLBaseFile,
)


class Module:
    def to_vhdl(self) -> Iterable[VHDLModule]:
        yield from [
            VHDLBaseModule(
                name="network",
                files=(
                    VHDLBaseFile(name="network.vhd", code=["vhdl code", "more code"]),
                ),
            )
        ]
