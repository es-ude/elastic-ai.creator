from collections import OrderedDict
from typing import cast

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from torch.nn import Linear, ReLU, Sequential

import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir.attribute import AttributeMapping
from elasticai.creator.ir2torch import Ir2TorchTranslationPass, IrFactory
from elasticai.creator.ir2torch import get_default_converter as ir2torch_converter
from elasticai.creator.ir2torch.default_handlers import linear, relu
from elasticai.creator.torch2ir import get_default_converter as torch2ir_converter


def model():
    return Sequential(
        Linear(1, 2),
        ReLU(),
    )


def torch2ir_legacy(model):
    return torch2ir_converter().convert(model)


def to_native_python(data) -> dict:
    result = {}
    for name, p in data:
        result[name] = p.tolist() if hasattr(p, "tolist") else p
    return result


def torch2ir(model):
    implementations = list(torch2ir_converter()(model))
    ir_dict = {
        impl.name: ir.IrDeserializerLegacy(IrFactory()).deserialize_graph(
            cast(dict[str, AttributeMapping], impl.data)
        )
        for impl in implementations
    }
    root = ir_dict.pop("")
    registry = ir.Registry(**ir_dict)
    return root, registry


def ir2torch(ir, state=None):
    translate = Ir2TorchTranslationPass()
    translate.register()(linear)
    translate.register()(relu)
    return translate(ir[0], ir[1], state)


def ir2torch_legacy(ir, state=None):
    return ir2torch_converter().convert(ir, state)


@pytest.mark.parametrize(
    "ir2torch, torch2ir", [(ir2torch_legacy, torch2ir_legacy), (ir2torch, torch2ir)]
)
@given(st.tuples(st.integers(1, 10), st.integers(1, 10)), st.booleans())
def test_build_model_from_ir_and_state_dict(ir2torch, torch2ir, num_features, bias):
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
    ir = torch2ir_legacy(original)
    rebuilt = ir2torch_converter().convert(ir, state)
    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_create_parent_modules_for_implemantations_with_dots():
    original = Sequential(OrderedDict(top=Sequential(OrderedDict(nested=Linear(1, 1)))))
    ir = torch2ir_legacy(original)
    rebuilt = ir2torch_converter().convert(ir, original.state_dict())
    original_params = to_native_python(original.named_parameters())
    rebuilt_params = to_native_python(rebuilt.named_parameters())
    assert original_params == rebuilt_params


def test_perform_inference():
    original = model()
    ir = torch2ir_legacy(original)
    rebuilt = ir2torch_converter().convert(ir)
    print(rebuilt.graph)
    rebuilt(torch.randn((1, 1)))
