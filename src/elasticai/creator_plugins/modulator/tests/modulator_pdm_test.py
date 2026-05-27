import pytest
from copy import deepcopy

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
from elasticai.creator_plugins.modulator.tests.modulator_pdm_tb import cocotb_settings


#@pytest.mark.simulation
@pytest.mark.parametrize(
    ["order", "cntwidth"], [
        (4, 8)
    ]
)
def test_modulator_pdm_verilog(order: int, cntwidth: int):
    sets = deepcopy(cocotb_settings)
    sets['params'] = {
        "MOD_ORDER": order,
        "CNTWIDTH_CLK": cntwidth
    }
    run_cocotb_sim_for_src_dir(**sets)


if __name__ == '__main__':
    pytest.main([__file__])
