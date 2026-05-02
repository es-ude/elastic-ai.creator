from abc import abstractmethod

import pytest
from elasticai.creator_plugins.lutron_filter import DataGraph, reorder
from elasticai.creator_plugins.lutron_filter.rules import _ir
from elasticai.creator_plugins.lutron_filter.rules._reorder import (
    _make_constraint,
    _pattern_graph,
)

from elasticai.creator.graph.graph_rewriting import get_rewriteable_matches
from elasticai.creator.ir.datagraph_rewriting import StdPattern
from elasticai.creator.ir.registry import Registry


@pytest.fixture
def factory() -> _ir.IrFactory:
    return _ir.ir_factory


def original_without_scaling():
    original = _ir.sequential(
        "input",
        ("a", "conv1d"),
        ("b", "maxpool1d"),
        ("c", "batchnorm1d"),
        ("d", "binarize"),
        "output",
    )
    original = original.add_nodes(
        ("a", _ir.attribute(type="conv1d", parameters="my_params")),
        ("b", _ir.attribute(type="maxpool1d", stride=2)),
        (
            "c",
            _ir.attribute(implementation="batchnorm1d", type="batchnorm1d"),
        ),
    )
    return original, _ir.Registry(
        batchnorm1d=_ir.ir_factory.graph(
            _ir.attribute(
                type="batchnorm1d",
                num_features=2,
                parameters=_ir.attribute(weight=(1.0, 0.5)),
            ),
        )
    )


def test_detect_pattern():
    original = original_without_scaling()
    _empty_reg = Registry()
    _constraint = _make_constraint(_empty_reg)

    def constraint(a, b):
        return _constraint(_ir.wrap_node(a), _ir.wrap_node(b))

    pattern = StdPattern(
        graph=_pattern_graph(), node_constraint=constraint, interface={"start", "end"}
    )
    matches = pattern.match(*original)
    matches = list(get_rewriteable_matches(original[0], matches, {"start", "end"}))
    expected_match = dict(
        start="a", maxpool1d="b", batchnorm1d="c", binarize="d", end="output"
    )
    assert matches[0] == expected_match
    assert len(matches) == 1


class TestReorderWithoutScaling:
    def _make_original(self) -> tuple[DataGraph, Registry]:
        return original_without_scaling()

    def _make_expected(self) -> tuple[DataGraph, Registry]:
        return _ir.sequential(
            "input", ("a", "conv1d"), "batchnorm1d", "binarize", "maxpool1d", "output"
        ).add_nodes(
            ("a", _ir.attribute(type="conv1d", parameters="my_params")),
            ("maxpool1d", _ir.attribute(type="maxpool1d", stride=2)),
            (
                "batchnorm1d",
                _ir.attribute(implementation="batchnorm1d", type="batchnorm1d"),
            ),
        ), _ir.Registry(
            batchnorm1d=_ir.ir_factory.graph(
                _ir.attribute(
                    type="batchnorm1d",
                    num_features=2,
                    parameters=_ir.attribute(weight=(1.0, 0.5)),
                ),
            )
        )

    def _make_result(self) -> tuple[DataGraph, Registry]:
        return reorder(*self.original)

    def _get_and_create_if_not_present(self, name, fn):
        if not hasattr(self, name):
            setattr(self, name, fn())
        return getattr(self, name)

    @property
    def original(self) -> tuple[DataGraph, Registry]:
        return self._get_and_create_if_not_present("_original", self._make_original)

    @property
    def expected(self) -> tuple[DataGraph, Registry]:
        return self._get_and_create_if_not_present("_expected", self._make_expected)

    @property
    @abstractmethod
    def result(self) -> tuple[DataGraph, Registry]:
        return self._get_and_create_if_not_present("_result", self._make_result)

    @property
    def expected_graph(self) -> DataGraph:
        return self.expected[0]

    @property
    def result_graph(self) -> DataGraph:
        return self.result[0]

    def test_has_correct_attributes(self) -> None:
        assert dict(self.result_graph.attributes) == dict(
            self.expected_graph.attributes
        )

    def test_has_correct_node_attributes(self) -> None:
        assert dict(self.result_graph.node_attributes) == dict(
            self.expected_graph.node_attributes
        )

    def test_has_correct_edges(self) -> None:
        assert set(self.result_graph.edges.keys()) == set(
            self.expected_graph.edges.keys()
        )

    def test_has_correct_registry(self) -> None:
        def to_dict(reg):
            return {name: dict(g.attributes.items()) for name, g in reg.items()}

        assert to_dict(self.result[1]) == to_dict(self.expected[1])


class TestReorderWithScaling(TestReorderWithoutScaling):
    def _make_original(
        self,
    ) -> tuple[DataGraph, Registry]:
        original, _ = super()._make_original()
        return original, _ir.Registry(
            batchnorm1d=_ir.ir_factory.graph(
                _ir.attribute(
                    type="batchnorm1d",
                    num_features=2,
                    parameters=_ir.attribute(weight=(-0.5, 0.5)),
                )
            )
        )

    def _make_expected(self) -> tuple[DataGraph, Registry]:
        scaling_compensation_attr = _ir.attribute(
            type="conv1d",
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            groups=2,
            in_channels=2,
            out_channels=2,
            parameters=_ir.attribute(weight=(((-1.0,),), ((1.0,),)), bias=(0.0, 0.0)),
        )
        scaling_compensation: Registry = _ir.Registry(
            scaling_compensation=_ir.sequential().with_attributes(
                scaling_compensation_attr
            ),
        )
        graph, registry = super()._make_expected()
        old_bnorm = registry["batchnorm1d"]
        registry = registry | _ir.Registry(
            batchnorm1d=old_bnorm.with_attributes(
                old_bnorm.attributes.update_path(("parameters", "weight"), (-0.5, 0.5))
            )
        )
        graph = graph.remove_edge("binarize", "maxpool1d").remove_edge(
            "maxpool1d", "output"
        )
        registry = registry | scaling_compensation

        return (
            graph.add_edges(
                ("binarize", "scaling_compensationA"),
                ("scaling_compensationA", "maxpool1d"),
                ("maxpool1d", "scaling_compensationB"),
                ("scaling_compensationB", "output"),
            ).add_nodes(
                (
                    "batchnorm1d",
                    _ir.attribute(implementation="batchnorm1d", type="batchnorm1d"),
                ),
                (
                    "scaling_compensationA",
                    _ir.attribute(type="conv1d", implementation="scaling_compensation"),
                ),
                (
                    "scaling_compensationB",
                    _ir.attribute(type="conv1d", implementation="scaling_compensation"),
                ),
            )
        ), registry
