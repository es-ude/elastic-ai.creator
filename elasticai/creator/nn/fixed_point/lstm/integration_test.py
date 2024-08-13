import pytest

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point import Linear
from elasticai.creator.nn.fixed_point.lstm.layer import (
    FixedPointLSTMWithHardActivations,
    LSTMNetwork,
)


@pytest.mark.simulation
def test_lstm_network_simulation(tmp_path):
    hidden_size = 20
    total_bits = 8
    frac_bits = 4
    model = LSTMNetwork(
        [
            FixedPointLSTMWithHardActivations(
                total_bits=total_bits,
                frac_bits=frac_bits,
                input_size=1,
                hidden_size=hidden_size,
                bias=True,
            ),
            Linear(
                total_bits=total_bits,
                frac_bits=frac_bits,
                in_features=hidden_size,
                out_features=1,
                bias=True,
            ),
        ]
    )
    design = model.create_design("lstm_network")
    testbench = model.create_testbench("lstm_network_tb", design)
    testbench.save_to(OnDiskPath(str(tmp_path), parent=""))
