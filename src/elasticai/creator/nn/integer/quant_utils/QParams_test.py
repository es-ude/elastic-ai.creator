import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricSignedQParams,
    AsymmetricUnsignedQParams,
    QuantScheme,
    SymmetricSignedQParams,
    SymmetricUnsignedQParams,
)


def test_AsymmetricSignedQParams():
    ASSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([[-2.0, 0.5, 1.0, 2.0, 3.0]], dtype=torch.float32)

    ASSSParams.update_quant_params(x)
    actual_q_x = ASSSParams.quantize(x)
    expected_q_x = torch.tensor([[-128, 0, 25, 76, 127]], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = ASSSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [[-2.0000, 0.5098, 1.0000, 2.0000, 3.0000]], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-10)


def test_AsymmetricUnsignedQParams():
    ASUSParams = AsymmetricUnsignedQParams(
        quant_bits=8, observer=GlobalMinMaxObserver()
    )

    x = torch.tensor([-2.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32)

    ASUSParams.update_quant_params(x)
    actual_q_x = ASUSParams.quantize(x)
    expected_q_x = torch.tensor([0, 128, 153, 204, 255], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = ASUSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [-2.0000, 0.5098, 1.0000, 2.0000, 3.0000], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


def test_SymmetricSignedQParams():
    SSSSParams = SymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())
    x = torch.tensor([-2.0, 0.5, 0.5, 1.0, 2.0], dtype=torch.float32)

    SSSSParams.update_quant_params(x)
    actual_q_x = SSSSParams.quantize(x)
    expected_q_x = torch.tensor([-127, 32, 32, 64, 127], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = SSSSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [-2.0000, 0.5039, 0.5039, 1.0079, 2.0000], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


# Observed range of x is [-2.0, 3.0]. Expected values are written out by hand:
# scale_factor = (max_float - min_float) / (max_quant - min_quant)
@pytest.mark.parametrize(
    "qparams_class, scale_factor, zero_point, min_float, max_float",
    [
        # [-2, 3] onto [-128, 127]: 5 / 255, zero point 127 - 3 / scale = -26
        (AsymmetricSignedQParams, 0.019607843831181526, -26, -2.0, 3.0),
        # [-2, 3] onto [0, 255]: 5 / 255, zero point 255 - 3 / scale = 102
        (AsymmetricUnsignedQParams, 0.019607843831181526, 102, -2.0, 3.0),
        # [-3, 3] onto [-127, 127]: 6 / 254, zero point 0
        (SymmetricSignedQParams, 0.023622047156095505, 0, -3.0, 3.0),
        # [0, 3] onto [0, 254]: 3 / 254, zero point 0
        (SymmetricUnsignedQParams, 0.011811023578047752, 0, 0.0, 3.0),
    ],
)
def test_update_quant_params(
    qparams_class, scale_factor, zero_point, min_float, max_float
):
    qparams = qparams_class(quant_bits=8, observer=GlobalMinMaxObserver())

    qparams.update_quant_params(
        torch.tensor([-2.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32)
    )

    assert torch.equal(
        qparams.scale_factor, torch.tensor([scale_factor], dtype=torch.float32)
    )
    assert torch.equal(
        qparams.zero_point, torch.tensor([zero_point], dtype=torch.int32)
    )
    assert torch.equal(qparams.min_float, torch.tensor([min_float]))
    assert torch.equal(qparams.max_float, torch.tensor([max_float]))


@pytest.mark.parametrize(
    "qparams_class, zero_point",
    [
        (AsymmetricSignedQParams, 127),
        (AsymmetricUnsignedQParams, 255),
        (SymmetricSignedQParams, 0),
        (SymmetricUnsignedQParams, 0),
    ],
)
def test_update_quant_params_of_constant_zero_uses_eps_as_scale_factor(
    qparams_class, zero_point
):
    qparams = qparams_class(quant_bits=8, observer=GlobalMinMaxObserver())

    qparams.update_quant_params(torch.zeros(3, dtype=torch.float32))

    assert torch.equal(
        qparams.scale_factor,
        torch.tensor([torch.finfo(torch.float32).eps], dtype=torch.float32),
    )
    assert torch.equal(
        qparams.zero_point, torch.tensor([zero_point], dtype=torch.int32)
    )


def test_set_scale_factor():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    scale_factor = torch.tensor([0.0157480314], dtype=torch.float32)
    SSSSParams.set_scale_factor(scale_factor)

    assert torch.equal(SSSSParams.scale_factor, scale_factor)


def test_set_zero_point():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    zero_point = torch.tensor([12], dtype=torch.int32)
    SSSSParams.set_zero_point(zero_point)

    assert torch.equal(SSSSParams.zero_point, zero_point)


def test_set_quant_range():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    SSSSParams.set_quant_range(16)

    assert SSSSParams.quant_bits == 16
    assert torch.equal(SSSSParams.min_quant, torch.tensor([-32768], dtype=torch.int32))
    assert torch.equal(SSSSParams.max_quant, torch.tensor([32767], dtype=torch.int32))


def test_quantize():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32)
    SSSSParams.update_quant_params(x)

    actual_q_x = SSSSParams.quantize(x)
    expected_q_x = torch.tensor([-128, 84, 127], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)


def test_dequantize():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32)
    SSSSParams.update_quant_params(x)
    q_x = SSSSParams.quantize(x)
    actual_dq_x = SSSSParams.dequantize(q_x)
    expected_dq_x = torch.tensor([-2.0000, 0.4941, 1.0000], dtype=torch.float32)
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


def test_state_dict():
    ASSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32)
    ASSSParams.update_quant_params(x)

    state_dict = ASSSParams.state_dict()

    ASSSParams_new = AsymmetricSignedQParams(
        quant_bits=8, observer=GlobalMinMaxObserver()
    )
    ASSSParams_new.load_state_dict(state_dict)

    assert torch.equal(ASSSParams.min_float, ASSSParams_new.min_float)
    assert torch.equal(ASSSParams.max_float, ASSSParams_new.max_float)
    assert torch.equal(ASSSParams.scale_factor, ASSSParams_new.scale_factor)
    assert torch.equal(ASSSParams.zero_point, ASSSParams_new.zero_point)
    assert torch.equal(ASSSParams.min_quant, ASSSParams_new.min_quant)
    assert torch.equal(ASSSParams.max_quant, ASSSParams_new.max_quant)


def test_str_separates_fields():
    qparams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    assert str(qparams) == (
        "min_float: 1.000000, max_float: 1.000000, min_quant: -128, "
        "max_quant: 127, quant_bits: 8, scale_factor: 1.0000000000, zero_point: 0"
    )


@pytest.mark.parametrize(
    "scheme, is_symmetric, is_signed",
    [
        (QuantScheme.ASYMMETRIC_SIGNED, False, True),
        (QuantScheme.SYMMETRIC_SIGNED, True, True),
        (QuantScheme.ASYMMETRIC_UNSIGNED, False, False),
        (QuantScheme.SYMMETRIC_UNSIGNED, True, False),
    ],
)
def test_QuantScheme_flags(scheme, is_symmetric, is_signed):
    assert scheme.is_symmetric is is_symmetric
    assert scheme.is_signed is is_signed


def test_subclass_uses_scheme():
    qparams = SymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    assert qparams.scheme is QuantScheme.SYMMETRIC_SIGNED
    assert qparams.is_symmetric
    assert qparams.is_signed


def test_state_dict_restores_quant_bits():
    qparams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())
    qparams.update_quant_params(torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32))
    qparams.set_quant_range(4)

    loaded = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())
    loaded.load_state_dict(qparams.state_dict())

    assert loaded.quant_bits == 4
    assert torch.equal(loaded.min_quant, qparams.min_quant)
    assert torch.equal(loaded.max_quant, qparams.max_quant)


def test_loads_state_dict_with_quant_bits_tensor():
    # State dicts of the former MPQParams.QParams store quant_bits as a tensor.
    qparams = AsymmetricSignedQParams(quant_bits=4, observer=GlobalMinMaxObserver())
    state_dict = qparams.state_dict()
    del state_dict["_extra_state"]
    state_dict["quant_bits"] = torch.tensor([4], dtype=torch.int32)

    loaded = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())
    loaded.load_state_dict(state_dict)

    assert loaded.quant_bits == 4
    assert torch.equal(loaded.max_quant, torch.tensor([7], dtype=torch.int32))


ALL_QPARAMS_CLASSES = [
    AsymmetricSignedQParams,
    AsymmetricUnsignedQParams,
    SymmetricSignedQParams,
    SymmetricUnsignedQParams,
]

quant_bits_strategy = st.integers(min_value=2, max_value=16)


@st.composite
def float_tensors(draw, non_negative: bool = False, contains_zero: bool = False):
    """Float32 tensors with finite values; optionally non-negative or containing 0."""
    elements = st.floats(
        min_value=0.0 if non_negative else -1e3,
        max_value=1e3,
        width=32,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
    values = draw(st.lists(elements, min_size=1, max_size=16))
    if contains_zero:
        values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def _expected_quant_range(qparams_class, quant_bits):
    return {
        AsymmetricSignedQParams: (-(2 ** (quant_bits - 1)), 2 ** (quant_bits - 1) - 1),
        SymmetricSignedQParams: (
            -(2 ** (quant_bits - 1)) + 1,
            2 ** (quant_bits - 1) - 1,
        ),
        AsymmetricUnsignedQParams: (0, 2**quant_bits - 1),
        SymmetricUnsignedQParams: (0, 2**quant_bits - 2),
    }[qparams_class]


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@given(quant_bits=quant_bits_strategy)
def test_quant_range_follows_quant_bits_and_scheme(qparams_class, quant_bits):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())

    expected_min, expected_max = _expected_quant_range(qparams_class, quant_bits)
    assert qparams.min_quant.item() == expected_min
    assert qparams.max_quant.item() == expected_max


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@given(quant_bits=quant_bits_strategy, new_quant_bits=quant_bits_strategy)
def test_set_quant_range_equals_construction_with_new_bits(
    qparams_class, quant_bits, new_quant_bits
):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())
    qparams.set_quant_range(new_quant_bits)

    expected_min, expected_max = _expected_quant_range(qparams_class, new_quant_bits)
    assert qparams.quant_bits == new_quant_bits
    assert qparams.min_quant.item() == expected_min
    assert qparams.max_quant.item() == expected_max


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@settings(deadline=None)
@given(quant_bits=quant_bits_strategy, x=float_tensors(non_negative=True))
def test_update_quant_params_invariants(qparams_class, quant_bits, x):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())

    qparams.update_quant_params(x)

    assert qparams.scale_factor.dtype == torch.float32
    assert qparams.zero_point.dtype == torch.int32
    assert qparams.scale_factor.item() >= torch.finfo(torch.float32).eps
    assert qparams.min_quant.item() <= qparams.zero_point.item()
    assert qparams.zero_point.item() <= qparams.max_quant.item()
    assert qparams.min_float.item() <= qparams.max_float.item()
    if qparams.is_symmetric:
        assert qparams.zero_point.item() == 0


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@settings(deadline=None)
@given(quant_bits=quant_bits_strategy, x=float_tensors(non_negative=True))
def test_quantize_stays_in_quant_range(qparams_class, quant_bits, x):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())
    qparams.update_quant_params(x)

    # values beyond the observed range must saturate, not wrap around
    x_q = qparams.quantize(torch.cat([x, 2 * x, -x]))

    assert x_q.dtype == torch.int32
    assert x_q.min().item() >= qparams.min_quant.item()
    assert x_q.max().item() <= qparams.max_quant.item()


@pytest.mark.parametrize(
    "qparams_class",
    [AsymmetricSignedQParams, AsymmetricUnsignedQParams, SymmetricSignedQParams],
)
@settings(deadline=None)
@given(quant_bits=quant_bits_strategy, x=float_tensors(contains_zero=True))
def test_roundtrip_error_is_at_most_half_a_step(qparams_class, quant_bits, x):
    # The observed range contains 0, otherwise the zero point is clamped and the
    # range is no longer fully representable.
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())
    qparams.update_quant_params(x)

    x_dq = qparams.dequantize(qparams.quantize(x))

    step = qparams.scale_factor.item()
    tolerance = step / 2 + 1e-5 * (1 + x.abs().max().item())
    assert (x_dq - x).abs().max().item() <= tolerance


@settings(deadline=None)
@given(quant_bits=quant_bits_strategy, x=float_tensors(non_negative=True))
def test_symmetric_unsigned_roundtrip_error_is_at_most_half_a_step(quant_bits, x):
    qparams = SymmetricUnsignedQParams(
        quant_bits=quant_bits, observer=GlobalMinMaxObserver()
    )
    qparams.update_quant_params(x)

    x_dq = qparams.dequantize(qparams.quantize(x))

    step = qparams.scale_factor.item()
    tolerance = step / 2 + 1e-5 * (1 + x.abs().max().item())
    assert (x_dq - x).abs().max().item() <= tolerance


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@settings(deadline=None)
@given(quant_bits=quant_bits_strategy, x=float_tensors(non_negative=True))
def test_quantize_is_idempotent_on_dequantized_values(qparams_class, quant_bits, x):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())
    qparams.update_quant_params(x)

    x_q = qparams.quantize(x)
    x_q_again = qparams.quantize(qparams.dequantize(x_q))

    assert torch.equal(x_q, x_q_again)


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@settings(deadline=None)
@given(quant_bits=quant_bits_strategy, x=float_tensors(non_negative=True))
def test_update_quant_params_is_independent_of_repetition(qparams_class, quant_bits, x):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())

    qparams.update_quant_params(x)
    scale_factor = qparams.scale_factor.clone()
    zero_point = qparams.zero_point.clone()
    qparams.update_quant_params(x)

    assert torch.equal(qparams.scale_factor, scale_factor)
    assert torch.equal(qparams.zero_point, zero_point)


@pytest.mark.parametrize("qparams_class", ALL_QPARAMS_CLASSES)
@given(quant_bits=quant_bits_strategy)
def test_state_dict_roundtrip_preserves_quant_bits(qparams_class, quant_bits):
    qparams = qparams_class(quant_bits=quant_bits, observer=GlobalMinMaxObserver())
    loaded = qparams_class(quant_bits=8, observer=GlobalMinMaxObserver())

    loaded.load_state_dict(qparams.state_dict())

    assert loaded.quant_bits == quant_bits


def test_SymmetricUnsignedQParams_quantizes_negative_values_to_zero():
    # Intended: the scheme has zero point 0 and range [0, 254], so it can only
    # represent non-negative values. Negative values saturate at 0.
    qparams = SymmetricUnsignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    qparams.update_quant_params(torch.tensor([0.0, 3.0], dtype=torch.float32))
    x = torch.tensor([-2.0, -0.1, 0.0, 3.0], dtype=torch.float32)
    x_q = qparams.quantize(x)

    assert torch.equal(x_q, torch.tensor([0, 0, 0, 254], dtype=torch.int32))
    assert torch.equal(qparams.dequantize(x_q)[:3], torch.zeros(3, dtype=torch.float32))
