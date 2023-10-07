from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.fixed_point import Linear
from elasticai.creator.nn.fixed_point.lstm.layer import (
    FixedPointLSTMWithHardActivations,
    LSTMNetwork,
)
from elasticai.creator.vhdl.system_integrations.plug_and_play_solution_ENV5 import (
    FirmwareENv5,
)

if __name__ == "__main__":
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
    destination = OnDiskPath("build")
    firmware = FirmwareENv5(model)
    firmware.save_to(destination)
