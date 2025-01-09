from elasticai.creator.lowering_pass_plugin import Lowerable, type_handler

from .convolution import DummyLowerable


@type_handler
def lowering_pass_conv(x: DummyLowerable) -> str:
    return "_".join(x.more_data)
