from collections.abc import Callable, Iterable, Iterator
from itertools import batched
from pathlib import Path

import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import elasticai.creator.function_dispatch as FD
import elasticai.creator.ir as ir
from elasticai.creator import ir2vhdl
from elasticai.creator.testing import ResetControl, StreamInterface
from elasticai.creator.testing.cocotb_pytest import CocotbTestFixture
from elasticai.creator_plugins.grouped_filter import FilterParameters, grouped_filter

type DataGraph = ir.DataGraph[ir.Node, ir.Edge]
type Registry = ir.Registry[DataGraph]

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


def create_ir2ir_translator():
    factory = ir2vhdl.IrFactory()

    def get_type(graph: DataGraph, registry: Registry):
        return graph.attributes["type"]

    def get_fn_name(fn):
        return fn.__name__

    @FD.create_keyed_dispatch(get_type, get_fn_name)
    def ir2ir(
        fn: Callable[[DataGraph, Registry], tuple[DataGraph, Registry]],
        graph: DataGraph,
        registry: Registry,
    ) -> tuple[DataGraph, Registry]:
        return fn(graph, registry)

    def lutron(graph: DataGraph, registry: Registry) -> tuple[DataGraph, Registry]:
        return graph, ir.Registry()

    def network(graph: DataGraph, registry: Registry) -> tuple[DataGraph, Registry]:
        g = factory.graph(type="clocked_combinatorial", name="network")

        def visit_top_down_depth_first(
            n: str, visited: set[str] | None = None
        ) -> Iterator[str]:
            if visited is None:
                visited = set()
            visited.add(n)
            for succ in graph.successors[n]:
                if succ not in visited:
                    yield succ
                    yield from visit_top_down_depth_first(succ, visited)

        g = g.add_node(graph.nodes["input"])
        prev_node = g.nodes["input"]

        def add_node(node: ir.Node):
            nonlocal g, prev_node
            g = g.add_node(node)
            g = g.add_edge(prev_node.name, node.name)
            prev_node = g.nodes[node.name]

        stride = 1

        def get_filter_params(node):
            return FilterParameters.from_dict(
                registry[node.attributes["implementation"]].attributes[
                    "filter_parameters"
                ]
            )

        for node_id in visit_top_down_depth_first("input"):
            node = graph.nodes[node_id]

            match node.attributes["type"]:
                case "input":
                    add_node(node)
                case "output":
                    add_node(
                        factory.node(
                            node.name,
                            node.attributes.new_with(
                                input_shape=prev_node.output_shape.to_tuple(),
                                output_shape=prev_node.output_shape.to_tuple(),
                            ),
                        ),
                    )

                case "grouped_filter":
                    params: FilterParameters = get_filter_params(node)
                    add_node(
                        factory.node(
                            f"{node_id}_sr",
                            ir.attribute(skip=stride),
                            type="shift_register",
                            implementation="shift_register",
                            input_shape=prev_node.output_shape,
                            output_shape=ir2vhdl.Shape(
                                params.in_channels, params.kernel_size
                            ),
                        )
                    )
                    stride = params.stride
                    add_node(
                        factory.node(
                            node_id,
                            type="unclocked_combinatorial",
                            implementation=f"{node.attributes['implementation']}",
                            input_shape=prev_node.output_shape,
                            output_shape=ir2vhdl.Shape(params.out_channels, 1),
                        )
                    )
                case _:
                    raise NotImplementedError()
        reg = {}
        for name, dgraph in registry.items():
            if dgraph is not graph:
                dgraph, subreg = ir2ir(dgraph, registry)
                reg[name] = dgraph
                reg |= subreg

        return g, ir.Registry(**reg)

    ir2ir.register("grouped_filter", grouped_filter)
    ir2ir.register("lutron", lutron)
    ir2ir.register()(network)
    return ir2ir


def make_inputs() -> Iterator[str]:
    for i in range(2**4):
        yield f"{i:04b}"


def make_ab_tt(input: str) -> str:
    if input == "0101":
        return "1"
    return "0"


def make_dc_tt(input: str) -> str:
    if input == "0011":
        return "1"
    return "0"


def create_ir2vhdl_translator():
    to_vhdl = ir2vhdl.Ir2Vhdl()
    _loader = ir2vhdl.PluginLoader(to_vhdl)
    _loader.load_from_package("lutron")
    _loader.load_from_package("shift_register")
    _loader.load_from_package("combinatorial")
    return to_vhdl


def translate(graph: DataGraph, registry: Registry, build_dir: Path):
    ir2ir = create_ir2ir_translator()
    to_vhdl = create_ir2vhdl_translator()
    lowered_graph, lowered_reg = ir2ir(graph, registry)
    lowered_reg = lowered_reg.add("network", lowered_graph)
    lowered_reg = ir2vhdl.collect_transitive_implementation_closure(
        "network", lowered_reg
    )

    code = to_vhdl(lowered_graph, lowered_reg)

    def add_separators(lines):
        for line in lines:
            yield line
            yield "\n"

    for name, lines in code:
        with open(build_dir / name, "w") as f:
            f.writelines(add_separators(lines))


def build_ir() -> tuple[DataGraph, Registry]:
    factory = ir2vhdl.IrFactory()
    channels = 4
    network = (
        factory.graph(
            name="network",
            type="network",
        )
        .add_nodes(
            factory.node(
                "input",
                type="input",
                input_shape=ir2vhdl.Shape(channels, 2),
                output_shape=ir2vhdl.Shape(channels, 2),
            ),
            factory.node(
                "output",
                type="output",
                input_shape=ir2vhdl.Shape(2, 1),
                output_shape=ir2vhdl.Shape(2, 1),
            ),
            factory.node(
                "conv0",
                type="grouped_filter",
                implementation="conv0",
                input_shape=ir2vhdl.Shape(channels, 2),
                output_shape=ir2vhdl.Shape(2, 1),
            ),
            factory.node(
                "conv1",
                type="grouped_filter",
                implementation="conv1",
                input_shape=ir2vhdl.Shape(2, 1),
                output_shape=ir2vhdl.Shape(2, 1),
            ),
        )
        .add_edges(("input", "conv0"), ("conv0", "conv1"), ("conv1", "output"))
    )

    def gen_table(fn):
        return tuple((input, fn(input)) for input in make_inputs())

    lutron_ab = factory.graph(
        ir.attribute(
            input_size=4,
            output_size=1,
            truth_table=gen_table(make_ab_tt),
            name="lutron_ab",
            type="lutron",
        )
    )
    lutron_cd = factory.graph(
        ir.attribute(
            input_size=4,
            output_size=1,
            truth_table=gen_table(make_dc_tt),
            name="lutron_cd",
            type="lutron",
        )
    )
    conv1 = factory.graph(
        ir.attribute(
            type="grouped_filter",
            kernel_per_group=("identity",),
            filter_parameters=FilterParameters(
                kernel_size=1, in_channels=2, out_channels=2, groups=1
            ).as_dict(),
        )
    )
    conv0 = factory.graph(
        ir.attribute(
            type="grouped_filter",
            kernel_per_group=("lutron_ab", "lutron_cd"),
            filter_parameters=FilterParameters(
                kernel_size=2,
                in_channels=4,
                out_channels=2,
                groups=2,
            ).as_dict(),
        )
    )
    identity = factory.graph(
        ir.attribute(
            type="lutron",
            input_size=2,
            output_size=2,
            truth_table=(("00", "00"), ("01", "01"), ("10", "10"), ("11", "11")),
        )
    )
    return network, ir.Registry(
        (
            ("lutron_ab", lutron_ab),
            ("lutron_cd", lutron_cd),
            ("conv0", conv0),
            ("conv1", conv1),
            ("identity", identity),
        )
    )


def interleave_inputs(inputs: Iterable[str]) -> str:
    result: list[str] = []
    for time_step in batched(inputs, 2):
        result.append("".join(time_step))
        result.append("".join(time_step))
    return "".join(result)


@ctb.test
async def check_grouped_filter_behaviour(dut):
    inputs = [interleave_inputs(inputs) for inputs in make_inputs()]
    expected = [make_ab_tt(input) + make_dc_tt(input) for input in make_inputs()]
    num_steps = len(inputs)
    assert 16 == num_steps

    ctb.start_soon(Clock(dut.clk, period=10).start())
    reset = ResetControl.from_dut(dut)
    stream = StreamInterface.from_dut(dut)
    dut.src_valid.value = 0
    dut.dst_ready.value = 0
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await reset.reset_active_high()
    collect_task = ctb.start_soon(
        stream.collect_chunks(expected_count=num_steps, max_cycles=2 * num_steps)
    )
    await stream.drive_chunks(inputs)
    observed = await collect_task
    incorrect = {}
    for input, exp, actual in zip(inputs, expected, observed):
        if exp != actual:
            incorrect[input] = {"expected": exp, "actual": actual}
    assert incorrect == {}


@pytest.mark.simulation
def test_network(cocotb_test_fixture: CocotbTestFixture):
    graph, registry = build_ir()
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    ir2vhd_build_dir = artifact_dir / "vhdl"
    ir2vhd_build_dir.mkdir(exist_ok=True)
    translate(graph, registry, ir2vhd_build_dir)
    all_files = []
    for f in ir2vhd_build_dir.iterdir():
        if f.is_file() and f.name.endswith("vhd"):
            all_files.append(f)

    cocotb_test_fixture.set_srcs(all_files)
    cocotb_test_fixture.run(params={}, defines={})
