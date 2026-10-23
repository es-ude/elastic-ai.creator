import importlib

from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


def test_imports_functions_not_modules():
    # Importing the package first runs `MPQSupport.py` while the package is
    # still being initialized, which is where a submodule could be bound.
    importlib.import_module("elasticai.creator.nn.integer.quant_utils")
    module = importlib.import_module(
        "elasticai.creator.nn.integer.quant_utils.MPQSupport"
    )

    assert module.scaling_M is scaling_M
    assert module.simulate_bitshifting is simulate_bitshifting
