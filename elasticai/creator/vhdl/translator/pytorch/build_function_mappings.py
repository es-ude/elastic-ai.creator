from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions import build_lstm_cell

DEFAULT_BUILD_FUNCTION_MAPPING = BuildFunctionMapping(
    mapping={"torch.nn.modules.rnn.LSTMCell": build_lstm_cell}
)
