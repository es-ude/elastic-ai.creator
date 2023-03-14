from elasticai.creator.vhdl.templates import VHDLTemplate


class DualPort2ClockRamComponent:
    def __init__(self, layer_id: str):
        self._name = "dual_port_2_clock_ram"
        self.template = VHDLTemplate(
            base_name=self._name,
        )
        self.id = layer_id

    @property
    def name(self) -> str:
        return f"{self._name}_{self.id}.vhd"

    def lines(self) -> list[str]:
        return self.template.lines()
