from pathlib import Path
from typing import Self

from elasticai.creator import ir
from elasticai.creator.ir2vhdl import (
    Ir2Vhdl,
    Node,
    PluginLoader,
    Registry,
    Shape,
    factory,
)
from elasticai.creator_plugins.skeleton.hw_function_id import HwFunctionIdUpdater


class CNNBuilder:
    def __init__(self, data_out_depth: int):
        if data_out_depth <= 0:
            raise ValueError("data_out_depth must be > 0")
        self._factory = factory
        self._root = self._factory.graph()
        self._registry: Registry = ir.Registry()
        self._last_node = self._factory.node("<none>")
        self._weights: list[str] = []
        self._data_width = 1
        self._data_out_depth = data_out_depth

    def add_conv(self, weight: str) -> Self:
        self._weights.append(weight)
        return self

    def _add_conv_to_root(self, weight: str, index: int) -> None:
        kernel_size = len(weight)
        factory = self._factory
        channels = 1

        shift = factory.node(
            f"sr{index}",
            type="shift_register",
            implementation="shift_register",
            input_shape=Shape(channels, self._data_width),
            output_shape=Shape(channels, kernel_size * self._data_width),
        )
        conv = factory.node(
            f"conv{index}",
            type="clocked_combinatorial",
            implementation=f"conv{index}",
            input_shape=Shape(channels, kernel_size * self._data_width),
            output_shape=Shape(channels, self._data_width),
        )
        self._update_root(
            nodes=[shift, conv],
            edges=[(self._last_node.name, shift.name), (shift.name, conv.name)],
        )

        self._last_node = self._root.nodes[conv.name]

    def _update_root(
        self, *, nodes: list[Node] = [], edges: list[tuple[str, str]] = []
    ) -> None:
        self._root = self._root.add_nodes(*nodes).add_edges(*edges)

    def _add_conv_to_reg(self, weight: str, index: int) -> None:
        kernel_size = len(weight)
        f = self._factory
        g = f.graph(
            ir.attribute(weight=f'"{weight}"', kernel_size=kernel_size, parallelism=1),
            name=self._last_node.name,
            type="binary_filter",
        )
        self._registry = self._registry | {g.name: g}

    def build(self) -> Registry:
        self._registry = ir.Registry()

        kernel_size = len(self._weights[0]) if self._weights else 1
        num_layers = len(self._weights)

        if num_layers == 0:
            raise ValueError("at least one convolution weight must be added")

        data_out_depth = self._data_out_depth
        data_in_depth = data_out_depth + num_layers * (kernel_size - 1)

        if data_in_depth <= 0:
            raise ValueError(f"data_in_depth must be positive, got {data_in_depth}")

        self._root = self._factory.graph(name="network", type="clocked_combinatorial")
        self._last_node = self._factory.node("<none>")

        self._update_root(
            nodes=[
                self._factory.node(
                    "input",
                    type="input",
                    input_shape=Shape(1, self._data_width),
                    output_shape=Shape(1, self._data_width),
                ),
                self._factory.node(
                    "output",
                    type="output",
                    input_shape=Shape(1, self._data_width),
                    output_shape=Shape(1, self._data_width),
                ),
            ]
        )

        self._last_node = self._root.nodes["input"]

        for i, weight in enumerate(self._weights):
            self._add_conv_to_root(weight, i)
            self._add_conv_to_reg(weight, i)

        self._update_root(edges=[(self._last_node.name, "output")])

        skeleton = self._factory.graph(
            ir.attribute(
                generic_map={
                    "DATA_IN_WIDTH": str(self._data_width),
                    "DATA_IN_DEPTH": str(data_in_depth),
                    "DATA_OUT_WIDTH": str(self._data_width),
                    "DATA_OUT_DEPTH": str(data_out_depth),
                }
            ),
            type="skeleton",
            name="skeleton",
        )

        self._registry = self._registry | {"network": self._root, "skeleton": skeleton}
        return self._registry


def prepare_ir2vhdl_for_sim():
    translator = Ir2Vhdl()
    plugins = PluginLoader(translator)
    plugins.load_from_package("skeleton")
    plugins.load_from_package("shift_register")
    plugins.load_from_package("combinatorial")
    plugins.load_from_package("xnor_popcount_mac")
    return translator


def build_network(
    build_dir: Path,
    weight: str = "10",
    data_in_depth: int = 4,
    ir2vhdl_fn=prepare_ir2vhdl_for_sim,
) -> bytes:
    ir2vhdl = ir2vhdl_fn()
    data_width = 1
    kernel_size = len(weight)
    network = (
        factory.graph(name="network", type="clocked_combinatorial")
        .add_nodes(
            factory.node(
                "input",
                type="input",
                input_shape=Shape(1, data_width),
                output_shape=Shape(1, data_width),
            ),
            factory.node(
                "output",
                type="output",
                input_shape=Shape(1, data_width),
                output_shape=Shape(1, data_width),
            ),
            factory.node(
                "sr0",
                type="shift_register",
                implementation="shift_register",
                input_shape=Shape(1, data_width),
                output_shape=Shape(1, kernel_size * data_width),
            ),
            factory.node(
                "conv1",
                type="clocked_combinatorial",
                implementation="conv1",
                input_shape=Shape(1, kernel_size * data_width),
                output_shape=Shape(1, data_width),
            ),
        )
        .add_edges(("input", "sr0"), ("sr0", "conv1"), ("conv1", "output"))
    )

    conv1 = factory.graph(
        ir.attribute(weight=f'"{weight}"', kernel_size=kernel_size, parallelism=1),
        name="conv1",
        type="binary_filter",
    )
    skeleton = factory.graph(
        ir.attribute(
            generic_map={
                "DATA_IN_WIDTH": str(data_width),
                "DATA_IN_DEPTH": str(data_in_depth),
                "DATA_OUT_WIDTH": str(data_width),
                "DATA_OUT_DEPTH": str(data_in_depth - kernel_size + 1),
            }
        ),
        type="skeleton",
        name="skeleton",
    )
    build_dir: Path = build_dir / "vhdl"
    if not build_dir.exists():
        build_dir.mkdir()
    hw_id = build_design(
        network,
        ir.Registry(conv1=conv1, network=network, skeleton=skeleton),
        build_dir,
        ir2vhdl,
    )
    return hw_id


def build_design(graph, registry, build_dir: Path, ir2vhdl) -> bytes:
    code = ir2vhdl(graph, registry)
    for filename, lines in code:
        (build_dir / filename).write_text("\n".join(lines) + "\n")
    hwid_updater = HwFunctionIdUpdater(build_dir)
    hwid_updater.compute_id()
    hwid_updater.write_id()
    return hwid_updater.id


def prepare_ir2vhdl_for_hw():

    ir2vhdl = prepare_ir2vhdl_for_sim()
    plugins = PluginLoader(ir2vhdl)
    plugins.load_from_package("middleware")
    return ir2vhdl


def collect_all_srcs_from_build_dir(build_dir):
    all_files = []
    for f in build_dir.iterdir():
        if f.is_file() and f.name.endswith("vhd"):
            all_files.append(f)
    return all_files
