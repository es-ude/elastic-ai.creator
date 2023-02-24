from elasticai.creator.vhdl.templates import VHDLTemplate


class NetworkComponent:
    def __init__(self):
        self.name = "network.vhd"
        self.template = VHDLTemplate(base_name="network")

    def lines(self) -> list[str]:
        return self.template.lines()
