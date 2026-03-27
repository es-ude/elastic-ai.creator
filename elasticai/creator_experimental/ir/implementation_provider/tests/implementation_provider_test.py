import pytest

from elasticai.creator import ir
from elasticai.creator_experimental.ir.implementation_provider.implementation_provider import (
    apply_training_provider,
)

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]


def _make_graph(**attrs) -> DataGraph:
    return ir.DefaultIrFactory().graph(ir.attribute(**attrs))


class MockFloat32Provider:
    def model_attributes(self, sub_graph: DataGraph) -> ir.AttributeMapping:
        return ir.attribute(quant_type="float32")

    def training_function(self) -> DataGraph:
        return _make_graph(type="training_function", step_fn="sgd_step")

    def optimizer(self) -> DataGraph:
        return _make_graph(type="optimizer", algorithm="sgd", lr=0.01)

    def loss(self) -> DataGraph:
        return _make_graph(type="loss", function="cross_entropy")


class FailingProvider(MockFloat32Provider):
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


def test_augments_model_registry_with_model_attributes():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32Provider()

    _, full_registry = apply_training_provider(model_root, model_registry, provider)

    for key in model_registry:
        assert full_registry[key].attributes.get("quant_type") == "float32"


def test_preserves_existing_model_attributes():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32Provider()

    _, full_registry = apply_training_provider(model_root, model_registry, provider)

    for key in model_registry:
        original = model_registry[key]
        for attr_key in original.attributes:
            assert full_registry[key].attributes.get(
                attr_key
            ) == original.attributes.get(attr_key)


def test_does_not_mutate_original_registry():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32Provider()

    apply_training_provider(model_root, model_registry, provider)

    for key, sub_graph in model_registry.items():
        assert "quant_type" not in sub_graph.attributes


def test_creates_training_program_with_all_component_nodes():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32Provider()

    training_program, _ = apply_training_provider(model_root, model_registry, provider)

    assert training_program.attributes.get("type") == "training_program"
    for name in ("model", "training_function", "optimizer", "loss"):
        assert name in training_program.nodes
        assert training_program.nodes[name].type == name


def test_registry_contains_model_and_provider_subgraphs():
    model_root, model_registry = _make_model_ir()
    provider = MockFloat32Provider()

    _, full_registry = apply_training_provider(model_root, model_registry, provider)

    assert full_registry["model"].attributes.get("type") == "module"
    assert full_registry["training_function"].attributes.get("step_fn") == "sgd_step"
    assert full_registry["optimizer"].attributes.get("algorithm") == "sgd"
    assert full_registry["optimizer"].attributes.get("lr") == 0.01
    assert full_registry["loss"].attributes.get("function") == "cross_entropy"


def test_raises_for_unregistered_node_type():
    model_root, model_registry = _make_model_ir()
    provider = FailingProvider()

    with pytest.raises(KeyError):
        apply_training_provider(model_root, model_registry, provider)


def test_works_with_empty_model_registry():
    model_root = _make_graph(type="module")
    model_registry: ir.Registry[DataGraph] = ir.Registry({})
    provider = MockFloat32Provider()

    training_program, full_registry = apply_training_provider(
        model_root, model_registry, provider
    )

    assert training_program.attributes.get("type") == "training_program"
    for name in ("model", "training_function", "optimizer", "loss"):
        assert name in full_registry


def test_raises_on_reserved_key_conflict():
    model_root = _make_graph(type="module")
    model_registry = ir.Registry({"model": _make_graph(type="linear")})
    provider = MockFloat32Provider()

    with pytest.raises(ValueError, match="reserved keys"):
        apply_training_provider(model_root, model_registry, provider)
