import dataclasses

import pytest

from elasticai.creator_plugins.quantized_grads.linear_quantization import (
    LinearQuantizationConfig,
)


def test_two_complement_fixed_point_config_is_frozen():
    config = LinearQuantizationConfig(num_bits=8)
    with pytest.raises(dataclasses.FrozenInstanceError):  #
        config.total_bits = 4
