from elasticai.creator.nn.vhdl import (
    FPHardSigmoid,
    FPHardTanh,
    FPLinear,
    FPReLU,
    FPSigmoid,
    Sequential,
)
from elasticai.creator.on_disk_path import OnDiskPath


def main() -> None:
    fp_args = dict(total_bits=16, frac_bits=8)
    model = Sequential(
        FPLinear(in_features=22, out_features=32, bias=True, **fp_args),
        FPReLU(total_bits=fp_args["total_bits"]),
        FPLinear(in_features=32, out_features=16, bias=True, **fp_args),
        FPHardSigmoid(**fp_args),
        FPLinear(in_features=16, out_features=8, bias=True, **fp_args),
        FPHardTanh(**fp_args),
        FPLinear(in_features=8, out_features=1, bias=True, **fp_args),
        FPSigmoid(**fp_args, num_steps=128, sampling_intervall=(-10, 10)),
    )

    destination = OnDiskPath("build", parent="./stuff")
    design = model.translate("fp_network")
    design.save_to(destination)


if __name__ == "__main__":
    main()
