from elasticai.creator.vhdl.vhdl_files import VHDLFile


class LSTMCommonVHDLFile(VHDLFile):
    def __init__(self, layer_id: str):
        super().__init__(name="lstm_common", layer_name=layer_id)
