import torch
from torch import Tensor
from elasticai.creator.vhdl.code_generation.code_abstractions import to_vhdl_binary_string
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import FixedPointConfig
from elasticai.creator.nn.fixed_point._math_operations import MathOperations


def tensor_to_vhdl_vector(x : Tensor, config : FixedPointConfig, as_matrix : bool = True) -> str:
    std_logic_vector: str = "(" if as_matrix else "\""
    for idx, val in enumerate(x):
        val = val.item()
        val = config.as_integer(val)
        std_logic_vector += to_vhdl_binary_string(val, config.total_bits) if as_matrix else to_vhdl_binary_string(val, config.total_bits).strip('\"')
        std_logic_vector += "," if as_matrix else ""
    std_logic_vector = std_logic_vector[:-1] if as_matrix else std_logic_vector
    std_logic_vector += ")" if as_matrix else "\""
    return std_logic_vector

if __name__ == "__main__":
    input = torch.linspace(0, 255, 255)
    print(f"{input=}")
    config = FixedPointConfig(total_bits=8, frac_bits=5)
    ops = MathOperations(config)
    input = input.apply_(lambda x: config.as_rational(x))
    input = torch.cat([input]*4, dim=0)
    input = input[0:914]
    print(f"{input=}")
    q_input = ops.quantize(input)
    print(f"{q_input=}")

    print(tensor_to_vhdl_vector(input, config=config, as_matrix=True))