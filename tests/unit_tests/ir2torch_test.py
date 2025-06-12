from collections import OrderedDict

import hypothesis.strategies as st
import torch
from hypothesis import given
from torch.nn import Linear, ReLU, Sequential

from elasticai.creator.ir2torch import get_default_converter as ir2torch_converter
from elasticai.creator.torch2ir import get_default_converter as torch2ir_converter


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def convert(model):
    return torch2ir_converter().convert(model)


def to_native_python(data) -> dict:
    result = {}
    for name, p in data:
        result[name] = p.tolist() if hasattr(p, "tolist") else p
    return result


@given(st.tuples(st.integers(1, 10), st.integers(1, 10)), st.booleans())
def test_build_model_from_ir_and_state_dict(num_features, bias):
    in_features, out_features = num_features
    original = Sequential(Linear(in_features, out_features, bias))
    ir = convert(original)
    state = original.state_dict()
    rebuilt = ir2torch_converter().convert(ir, state)
    rebuilt.load_state_dict(state)

    def to_native_python(data) -> dict:
        result = {}
        for name, p in data:
            result[name] = p.tolist() if hasattr(p, "tolist") else p
        return result

    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_can_rebuild_model_with_duplicate_submodule():
    lin = Linear(1, 1)
    original = Sequential(lin, lin)
    state = original.state_dict()
    ir = convert(original)
    rebuilt = ir2torch_converter().convert(ir, state)
    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_create_parent_modules_for_implemantations_with_dots():
    original = Sequential(OrderedDict(top=Sequential(OrderedDict(nested=Linear(1, 1)))))
    ir = convert(original)
    rebuilt = ir2torch_converter().convert(ir, original.state_dict())
    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_perform_inference():
    original = model()
    ir = convert(original)
    rebuilt = ir2torch_converter().convert(ir)
    print(rebuilt.graph)
    rebuilt(torch.randn((1, 1)))
