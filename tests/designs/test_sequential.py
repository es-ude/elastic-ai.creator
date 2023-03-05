import unittest

from elasticai.creator.hdl.code_generation.code_generation import (
    calculate_address_width,
)
from elasticai.creator.hdl.design_base import std_signals
from elasticai.creator.hdl.translatable import File, Path
from elasticai.creator.hdl.vhdl.code_generation.code_generation import (
    create_connections,
    create_instance,
    signal_definition,
)
from elasticai.creator.hdl.vhdl.code_generation.template import Template
from elasticai.creator.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn._two_complement_fixed_point_config import (
    TwoComplementFixedPointConfig,
)
from elasticai.creator.nn.linear import FixedPointConfig
from elasticai.creator.translatable_modules.vhdl.fp_hard_sigmoid import FPHardSigmoid
from elasticai.creator.translatable_modules.vhdl.fp_linear_1d import FPLinear1d
from elasticai.creator.translatable_modules.vhdl.sequential import Sequential


class SequentialTestCase(unittest.TestCase):
    """
    Tests:
      - add a layer with buffer
      - test each section (connections, instantiations, etc.) in isolation

    """

    def test_empty_sequential(self):
        module = Sequential(tuple())
        template = SequentialTemplate(
            connections=sorted(
                ["y <= x;", "x_address <= y_address;", "done <= enable;"]
            ),
            instantiations=[],
            signal_definitions=[],
            x_width="1",
            y_width="1",
            x_address_width="1",
            y_address_width="1",
        )
        expected = template.lines()
        destination = InMemoryPathForTesting("sequential")
        design = module.translate()
        design.save_to(destination)
        self.assertEqual(expected, destination.text)

    def test_autowire_instantiate_and_define_signals_for_hard_sigmoid_activation(self):
        bit_width = 16

        template = _prepare_sequential_template_with_hard_sigmoid(bit_width)
        expected = template.lines()

        module = Sequential((FPHardSigmoid(FixedPointConfiguration()),))
        design = module.translate()

        destination = InMemoryPathForTesting("sequential")
        design.save_to(destination)
        self.assertEqual(expected, destination.text)

    def test_with_single_linear_layer(self):
        bit_width = 16
        template = _prepare_sequential_template_with_linear(bit_width)
        expected = template.lines()
        module = Sequential(
            (
                (
                    FPLinear1d(
                        in_features=4,
                        out_features=2,
                        config=FixedPointConfig(total_bits=16, frac_bits=4),
                        bias=False,
                    ),
                )
            )
        )
        design = module.translate()
        destination = InMemoryPathForTesting("sequential")
        design.save_to(destination)
        self.assertEqual(expected, destination.text)


class SequentialTemplate:
    def __init__(
        self,
        connections: list[str],
        instantiations: list[str],
        signal_definitions: list[str],
        x_width: str,
        y_width: str,
        x_address_width: str,
        y_address_width: str,
    ):
        self._template = Template(
            "network",
            layer_connections=connections,
            layer_instantiations=instantiations,
            signal_definitions=signal_definitions,
            x_width=x_width,
            y_width=y_width,
            x_address_width=x_address_width,
            y_address_width=y_address_width,
        )

    def lines(self) -> list[str]:
        return self._template.lines()


class InMemoryPathForTesting(Path):
    def __init__(self, subpath_name: str):
        self._root = InMemoryPath("root", parent=None)
        self._subpath_name = subpath_name
        self._suffix = ""
        self._subpath = self._root.create_subpath(self._subpath_name)

    def create_subpath(self, subpath_name: str) -> "Path":
        return self._subpath.create_subpath(subpath_name)

    def as_file(self, suffix: str) -> File:
        self._suffix = suffix
        return self._subpath.as_file(suffix)

    @property
    def text(self) -> list[str]:
        child = self._root.children[f"{self._subpath_name}{self._suffix}"]
        assert isinstance(child, InMemoryFile)
        return child.text


def _prepare_sequential_template_with_linear(bit_width) -> SequentialTemplate:
    entity = "fp_linear1d"
    instance = f"i_{entity}_0"
    connections = create_connections(
        {
            f"{instance}_x": "x",
            "y": f"{instance}_y",
            f"{instance}_enable": "enable",
            f"{instance}_clock": "clock",
            "done": f"{instance}_done",
            f"{instance}_y_address": "y_address",
            "x_address": f"{instance}_x_address",
        }
    )

    instantiations = create_instance(
        name=instance,
        entity=entity,
        architecture="rtl",
        library="work",
        signal_mapping={
            s: f"{instance}_{s}"
            for s in ("x", "y", "clock", "enable", "x_address", "y_address", "done")
        },
    )
    address_width = calculate_address_width(bit_width)

    signal_definitions = sorted(
        [
            signal_definition(name=f"{instance}_{signal.name}", width=signal.width)
            for signal in (
                std_signals.x(bit_width),
                std_signals.y(bit_width),
                std_signals.clock(),
                std_signals.enable(),
                std_signals.done(),
                std_signals.x_address(address_width),
                std_signals.y_address(address_width),
            )
        ]
    )

    template = SequentialTemplate(
        connections=connections,
        instantiations=instantiations,
        signal_definitions=signal_definitions,
        x_width=f"{bit_width}",
        y_width=f"{bit_width}",
        x_address_width=str(calculate_address_width(bit_width)),
        y_address_width=str(calculate_address_width(bit_width)),
    )
    return template


def _prepare_sequential_template_with_hard_sigmoid(
    bit_width: int,
) -> SequentialTemplate:
    instance = "i_fp_hard_sigmoid_0"
    connections = create_connections(
        {
            f"{instance}_x": "x",
            "y": f"{instance}_y",
            "done": "enable",
            f"{instance}_clock": "clock",
            f"{instance}_enable": "enable",
            "x_address": "y_address",
        }
    )

    instantiations = create_instance(
        name=instance,
        entity="fp_hard_sigmoid",
        architecture="rtl",
        library="work",
        signal_mapping={s: f"{instance}_{s}" for s in ("x", "y", "clock", "enable")},
    )

    signal_definitions = sorted(
        [
            signal_definition(name=f"{instance}_{signal.name}", width=signal.width)
            for signal in (
                std_signals.x(bit_width),
                std_signals.y(bit_width),
                std_signals.clock(),
                std_signals.enable(),
            )
        ]
    )

    template = SequentialTemplate(
        connections=connections,
        instantiations=instantiations,
        signal_definitions=signal_definitions,
        x_width=f"{bit_width}",
        y_width=f"{bit_width}",
        x_address_width=str(calculate_address_width(bit_width)),
        y_address_width=str(calculate_address_width(bit_width)),
    )
    return template


class FixedPointConfiguration:
    def __init__(self):
        self.total_bits = 16
        self.frac_bits = 8


if __name__ == "__main__":
    unittest.main()
