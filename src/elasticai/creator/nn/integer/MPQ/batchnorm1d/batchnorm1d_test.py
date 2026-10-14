import pytest

from elasticai.creator.nn.integer.MPQ.batchnorm1d.batchnorm1d import BatchNorm1d


def create_batchnorm(**kwargs) -> BatchNorm1d:
    return BatchNorm1d(
        name="bn_0",
        norm_dim=3,
        eps=1e-5,
        momentum=0.1,
        track_running_stats=True,
        device="cpu",
        **kwargs,
    )


def test_affine_false_is_not_implemented():
    with pytest.raises(NotImplementedError):
        create_batchnorm(affine=False)


def test_is_affine_by_default():
    assert create_batchnorm().affine


def test_bias_quant_bits_are_derived_from_inputs_and_weights():
    layer = create_batchnorm()

    layer.set_quant_bits_from_config(
        {"bn_0.inputs": 6, "bn_0.weights": 4, "bn_0.bias": 3, "bn_0.outputs": 8}
    )

    assert layer.bias_QParams.quant_bits == (6 + 1) + (4 + 1)


def test_missing_quant_bits_in_config_raises():
    layer = create_batchnorm()

    with pytest.raises(ValueError, match="bn_0.weights"):
        layer.set_quant_bits_from_config({"bn_0.inputs": 8, "bn_0.outputs": 8})
