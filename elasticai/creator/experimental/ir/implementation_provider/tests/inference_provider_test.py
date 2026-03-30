import pytest

from elasticai.creator import ir
from elasticai.creator.experimental.ir.implementation_provider.implementation_provider import (
    apply_inference_provider,
)

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]


class MockFloat32InferenceProvider:
    def model_attributes(self, sub_graph: DataGraph) -> ir.AttributeMapping:
        return ir.attribute(quant_type="float32")


class FailingInferenceProvider:
    def model_attributes(self, sub_graph: DataGraph) -> ir.AttributeMapping:
        node_type = sub_graph.attributes["type"]
        if node_type == "linear":
            return ir.attribute(quant_type="float32")
        raise KeyError(f"No implementation for type '{node_type}'")


def _make_model_ir() -> tuple[DataGraph, ir.Registry[DataGraph]]:
    factory = ir.DefaultIrFactory()
    model_root = (
        factory.graph(ir.attribute(type="module"))
        .add_node("input_1", ir.attribute(type="input"))
        .add_node("linear_0", ir.attribute(type="linear", implementation="linear_0"))
        .add_node("relu_1", ir.attribute(type="relu", implementation="relu_1"))
        .add_node("output_1", ir.attribute(type="output"))
        .add_edge("input_1", "linear_0")
        .add_edge("linear_0", "relu_1")
        .add_edge("relu_1", "output_1")
    )
    model_registry = ir.Registry(
        {
            "linear_0": factory.graph(
                ir.attribute(type="linear", in_features=4, out_features=2)
            ),
            "relu_1": factory.graph(ir.attribute(type="relu")),
        }
    )
    return model_root, model_registry


def test_apply_inference_provider_augments_registry_with_model_attributes():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32InferenceProvider()

    _, augmented_registry = apply_inference_provider(
        model_root, model_registry, provider
    )

    for key in model_registry:
        assert augmented_registry[key].attributes.get("quant_type") == "float32"


def test_apply_inference_provider_preserves_existing_attributes():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32InferenceProvider()

    _, augmented_registry = apply_inference_provider(
        model_root, model_registry, provider
    )

    for key in model_registry:
        original = model_registry[key]
        for attr_key in original.attributes:
            assert augmented_registry[key].attributes.get(
                attr_key
            ) == original.attributes.get(attr_key)


def test_apply_inference_provider_does_not_mutate_original_registry():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32InferenceProvider()

    apply_inference_provider(model_root, model_registry, provider)

    for key, sub_graph in model_registry.items():
        assert "quant_type" not in sub_graph.attributes


def test_apply_inference_provider_returns_original_root_unchanged():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32InferenceProvider()

    returned_root, _ = apply_inference_provider(model_root, model_registry, provider)

    assert returned_root.attributes == model_root.attributes


def test_apply_inference_provider_raises_for_unregistered_node_type():
    model_root, model_registry = _make_model_ir()
    provider = FailingInferenceProvider()

    with pytest.raises(KeyError):
        apply_inference_provider(model_root, model_registry, provider)


def test_apply_inference_provider_works_with_empty_registry():
    factory = ir.DefaultIrFactory()
    model_root = factory.graph(ir.attribute(type="module"))
    model_registry: ir.Registry[DataGraph] = ir.Registry({})
    provider = MockFloat32InferenceProvider()

    returned_root, augmented_registry = apply_inference_provider(
        model_root, model_registry, provider
    )

    assert returned_root.attributes.get("type") == "module"
    assert len(augmented_registry) == 0
