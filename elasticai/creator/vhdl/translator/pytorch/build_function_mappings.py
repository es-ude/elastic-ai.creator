from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions import (
    build_linear_1d,
    build_lstm,
)

DEFAULT_BUILD_FUNCTION_MAPPING = BuildFunctionMapping(
    mapping={
        "torch.nn.modules.rnn.LSTM": build_lstm,
        "torch.nn.modules.linear.Linear": build_linear_1d,
    }
)
