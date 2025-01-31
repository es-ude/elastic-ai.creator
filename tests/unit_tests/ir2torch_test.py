import hypothesis.strategies as st
from hypothesis import given
from torch.nn import Linear, ReLU, Sequential

from elasticai.creator.ir2torch import Ir2Torch
from elasticai.creator.torch2ir import Torch2Ir


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def convert(model):
    return Torch2Ir.get_default_converter().convert(model)


@given(st.tuples(st.integers(1, 10), st.integers(1, 10)), st.booleans())
def test_build_model_from_ir_and_state_dict(num_features, bias):
    in_features, out_features = num_features
    original = Sequential(Linear(in_features, out_features, bias))
    ir = convert(original)
    state = original.state_dict()
    print(ir)
    rebuilt = Ir2Torch.get_default_converter().convert(ir)
    print(rebuilt)
    rebuilt.load_state_dict(state)

    def to_native_python(data) -> dict:
        result = {}
        for name, p in data:
            result[name] = p.tolist() if hasattr(p, "tolist") else p
        return result

    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params
