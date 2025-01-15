import dataclasses

import pytest

from .two_complement_fixed_point_config import FixedPointConfigV2


def test_two_complement_fixed_point_config_is_frozen():
    config = FixedPointConfigV2(total_bits=8, frac_bits=2)
    with pytest.raises(dataclasses.FrozenInstanceError):  #
        config.total_bits = 4
