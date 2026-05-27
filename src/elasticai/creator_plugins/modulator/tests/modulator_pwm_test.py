import pytest
from copy import deepcopy

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
from elasticai.creator_plugins.modulator.tests.modulator_pwm_tb import cocotb_settings


#@pytest.mark.simulation
@pytest.mark.parametrize(
    "period", [2, 4, 6, 8, 16, 32, 64, 128, 256, 1024, 4096]
)
def test_modulator_pwm_verilog(period: int):
    sets = deepcopy(cocotb_settings)
    sets['params'] = {"PERIOD_NUM_CYCLE": period}
    run_cocotb_sim_for_src_dir(**sets)


if __name__ == '__main__':
    pytest.main([__file__])
