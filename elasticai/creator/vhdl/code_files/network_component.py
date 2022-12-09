from elasticai.creator.vhdl.vhdl_files import VHDLFile


class NetworkVHDLFile(VHDLFile):
    def __init__(self):
        super().__init__(
            name="network",
        )
