import pytest
from copy import deepcopy

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
from elasticai.creator_plugins.mac.tests.uint_to_int_tb import cocotb_settings


#@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 4, 6, 8, 10, 12, 16])
def test_integer_transformation_verilog(bitwidth: int):
    set0 = deepcopy(cocotb_settings)
    set0['params'] = {'BITWIDTH': bitwidth}
    run_cocotb_sim_for_src_dir (**set0)


if __name__ == '__main__':
    pytest.main([__file__])
