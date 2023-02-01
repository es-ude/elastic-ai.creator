from elasticai.creator.vhdl.designs.vhdl_files import VHDLFile


class NetworkVHDLFile(VHDLFile):
    def __init__(self):
        super().__init__(
            template_name="network",
        )
