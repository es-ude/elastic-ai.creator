from collections import OrderedDict

import hypothesis.strategies as st
import torch
from hypothesis import given
from torch.nn import Linear, ReLU, Sequential

import elasticai.creator.torch2ir.default_handlers as t2ir_handlers
from elasticai.creator.ir2torch import Ir2TorchTranslationPass as Ir2Torch
from elasticai.creator.ir2torch.default_handlers import linear, relu
from elasticai.creator.torch2ir import Torch2IrTranslator as Torch2Ir


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def to_native_python(data) -> dict:
    result = {}
    for name, p in data:
        result[name] = p.tolist() if hasattr(p, "tolist") else p
    return result


def torch2ir(model):
    translate = Torch2Ir()
    for handler in (
        t2ir_handlers.batchnorm1d,
        t2ir_handlers.linear,
        t2ir_handlers.relu,
    ):
        translate.register()(handler)
    implementation, registry = translate(model)
    return implementation, registry


def ir2torch(ir, state=None):
    translate = Ir2Torch()
    translate.register()(linear)
    translate.register()(relu)
    return translate(ir[0], ir[1], state)


@given(st.tuples(st.integers(1, 10), st.integers(1, 10)), st.booleans())
def test_build_model_from_ir_and_state_dict(num_features, bias):
    in_features, out_features = num_features
    original = Sequential(Linear(in_features, out_features, bias))
    ir = torch2ir(original)
    state = original.state_dict()
    rebuilt = ir2torch(ir, state)
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
    ir = torch2ir(original)
    rebuilt = ir2torch(ir, state)
    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_create_parent_modules_for_implemantations_with_dots():
    original = Sequential(OrderedDict(top=Sequential(OrderedDict(nested=Linear(1, 1)))))
    ir = torch2ir(original)
    rebuilt = ir2torch(ir, original.state_dict())
    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_perform_inference():
    original = model()
    ir = torch2ir(original)
    rebuilt = ir2torch(ir)
    print(rebuilt.graph)
    rebuilt(torch.randn((1, 1)))
