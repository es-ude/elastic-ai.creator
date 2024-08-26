from pathlib import Path

from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, ReLU
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5


def main() -> None:
    inputs = 2
    outputs = 2
    total_bits = 8
    frac_bits = 4

    destination = Path("build_dir")

    model = Sequential(
        Linear(
            in_features=inputs,
            out_features=outputs,
            total_bits=total_bits,
            frac_bits=frac_bits,
        ),
        ReLU(total_bits=total_bits),
    )
    my_design = model.create_design("myNetwork")
    my_design.save_to(destination / "srcs")

    firmware = FirmwareENv5(
        network=my_design,
        x_num_values=inputs,
        y_num_values=outputs,
        id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        skeleton_version="v2",
    )
    firmware.save_to(destination)


if __name__ == "__main__":
    main()
