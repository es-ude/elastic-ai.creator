from vhdl.hw_equivalent_layers.vhdl_files import VHDLFile


class LSTMCommonVHDLFile(VHDLFile):
    def __init__(self):
        super().__init__(
            name="lstm_common",
        )
