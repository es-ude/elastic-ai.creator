from vhdl.hw_equivalent_layers.vhdl_files import VHDLFile


class DualPort2ClockRamVHDLFile(VHDLFile):
    def __init__(self):
        super().__init__(
            name="dual_port_2_clock_ram",
        )
