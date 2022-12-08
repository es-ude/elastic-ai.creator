from elasticai.creator.vhdl.vhdl_files import VHDLFile


class NetworkVHDLFile(VHDLFile):
    def save_to(self, prefix: str):
        raise NotImplementedError()

    def __init__(self):
        super().__init__(
            name="network",
        )
