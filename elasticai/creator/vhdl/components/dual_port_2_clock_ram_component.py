from elasticai.creator.vhdl.vhdl_files import StaticVHDLFile


class DualPort2ClockRamVHDLFile(StaticVHDLFile):
    def __init__(self):
        super().__init__(
            template_package="elasticai.creator.vhdl.templates",
            file_name="dual_port_2_clock_ram.vhd",
        )
