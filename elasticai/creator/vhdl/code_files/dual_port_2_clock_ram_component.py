from elasticai.creator.vhdl.language.vhdl_template import VHDLTemplate


class DualPort2ClockRamVHDLFile(VHDLTemplate):
    def __init__(self):
        super().__init__(
            template_name="dual_port_2_clock_ram",
        )
