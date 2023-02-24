from elasticai.creator.vhdl.templates import VHDLTemplate


class NetworkVHDLFile(VHDLTemplate):
    def __init__(self):
        super().__init__(
            base_name="network",
        )
