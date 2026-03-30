from abc import abstractmethod
from typing import Protocol

from elasticai.creator import ir

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]
type Registry = ir.Registry[DataGraph]


class TrainingImplementationProvider(Protocol):
    """Provider for training-specific implementation attributes.

    Backends (e.g., ODT) implement this protocol to declare model layer
    attributes, training function, optimizer, and loss for on-device training.
    """

    @abstractmethod
    def model_attributes(self, sub_graph: DataGraph) -> ir.AttributeMapping: ...

    @abstractmethod
    def training_function(self) -> DataGraph: ...

    @abstractmethod
    def optimizer(self) -> DataGraph: ...

    @abstractmethod
    def loss(self) -> DataGraph: ...


class InferenceImplementationProvider(Protocol):
    """Provider for inference-only implementation attributes.

    Backends implement this protocol to declare model layer attributes
    for inference without training (no optimizer, loss, or training loop).
    """

    @abstractmethod
    def model_attributes(self, sub_graph: DataGraph) -> ir.AttributeMapping: ...


def apply_inference_provider(
    model_root: DataGraph,
    model_registry: Registry,
    provider: InferenceImplementationProvider,
) -> tuple[DataGraph, Registry]:
    augmented: dict[str, DataGraph] = {}
    for key, sub_graph in model_registry.items():
        new_attrs = provider.model_attributes(sub_graph)
        augmented[key] = sub_graph.with_attributes(sub_graph.attributes | new_attrs)
    return model_root, ir.Registry(augmented)


_TRAINING_PROGRAM_COMPONENTS = ("model", "training_function", "optimizer", "loss")


def apply_training_provider(
    model_root: DataGraph,
    model_registry: Registry,
    provider: TrainingImplementationProvider,
) -> tuple[DataGraph, Registry]:
    conflicts = set(_TRAINING_PROGRAM_COMPONENTS) & set(model_registry.keys())
    if conflicts:
        raise ValueError(f"model_registry contains reserved keys: {conflicts}")

    augmented_model_entries: dict[str, DataGraph] = {}
    for key, sub_graph in model_registry.items():
        new_attrs = provider.model_attributes(sub_graph)
        augmented_model_entries[key] = sub_graph.with_attributes(
            sub_graph.attributes | new_attrs
        )

    factory = ir.DefaultIrFactory()
    training_program = factory.graph(ir.attribute(type="training_program"))
    for name in _TRAINING_PROGRAM_COMPONENTS:
        training_program = training_program.add_node(
            name, ir.attribute(type=name, implementation=name)
        )

    full_registry = ir.Registry(
        {
            **augmented_model_entries,
            "model": model_root,
            "training_function": provider.training_function(),
            "optimizer": provider.optimizer(),
            "loss": provider.loss(),
        }
    )

    return training_program, full_registry
