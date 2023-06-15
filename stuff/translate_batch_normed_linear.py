from elasticai.creator.nn.vhdl import FPBatchNormedLinear, Sequential
from elasticai.creator.on_disk_path import OnDiskPath


def main() -> None:
    model = Sequential(
        FPBatchNormedLinear(
            in_features=1,
            out_features=10,
            bias=True,
            total_bits=16,
            frac_bits=8,
            bn_affine=True,
        )
    )

    destination = OnDiskPath(name="build", parent="./stuff")
    design = model.translate("fp_network")
    design.save_to(destination)


if __name__ == "__main__":
    main()
