from elasticai.creator.vhdl.designs.vhdl_files import VHDLFile


class DualPort2ClockRamVHDLFile(VHDLFile):
    def __init__(self):
        super().__init__(
            template_name="dual_port_2_clock_ram",
        )
