from elasticai.creator.vhdl.translator.build_function_mapping import (
    BuildFunctionMapping,
)
from elasticai.creator.vhdl.translator.pytorch.build_functions import (
    build_fixed_point_lstm,
    build_fp_hard_sigmoid,
    build_fp_hard_tanh,
    build_fp_linear_1d,
    build_fp_relu,
)

DEFAULT_BUILD_FUNCTION_MAPPING = BuildFunctionMapping(
    mapping={
        "elasticai.creator.nn.lstm.FixedPointLSTMWithHardActivations": (
            build_fixed_point_lstm
        ),
        "elasticai.creator.nn.linear.FixedPointLinear": build_fp_linear_1d,
        "elasticai.creator.nn.hard_sigmoid.HardSigmoid": build_fp_hard_sigmoid,
        "elasticai.creator.nn.relu.ReLU": build_fp_relu,
        "elasticai.creator.nn.hard_tanh.HardTanh": build_fp_hard_tanh,
    }
)
