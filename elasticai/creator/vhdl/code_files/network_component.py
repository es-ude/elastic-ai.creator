from elasticai.creator.vhdl.templates.vhdl_template import VHDLTemplate


class NetworkVHDLFile(VHDLTemplate):
    def __init__(self):
        super().__init__(
            template_name="network",
        )
