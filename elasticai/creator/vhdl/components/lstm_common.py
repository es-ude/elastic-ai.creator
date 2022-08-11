from elasticai.creator.vhdl.vhdl_component import VHDLStaticComponent


class LSTMCommon(VHDLStaticComponent):
    def __init__(self):
        super().__init__(
            template_package="elasticai.creator.vhdl.templates",
            file_name="lstm_common.vhd",
        )
