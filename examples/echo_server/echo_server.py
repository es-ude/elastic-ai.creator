from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.system_integrations.plug_and_play_solution_ENV5 import (
    FirmwareEchoServerSkeletonV2,
)

f = FirmwareEchoServerSkeletonV2(num_inputs=4, bitwidth=8)
f.save_to(OnDiskPath("build"))
