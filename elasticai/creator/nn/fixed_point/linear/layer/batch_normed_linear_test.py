from .batch_normed_linear import BatchNormedLinear


def test_create_design() -> None:
    layer = BatchNormedLinear(in_features=4, out_features=2, total_bits=12, frac_bits=4)

    layer.create_design(name="linear")
