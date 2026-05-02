import elasticai.creator.ir as ir
from elasticai.creator.graph import bfs_iter_down

from ._ir import DataGraph


def make_add_bitwidths_rule(input_bitwidth: int) -> ir.Rule[DataGraph, DataGraph]:
    def rule(
        graph: DataGraph, reg: ir.Registry[DataGraph]
    ) -> tuple[DataGraph, ir.Registry[DataGraph]]:
        input_node = ""
        for n in graph.nodes.values():
            if n.type == "input":
                input_node = n.name

        for node in bfs_iter_down(
            graph.successors.get,  # type: ignore
            graph.predecessors.get,  # type: ignore
            input_node,
        ):
            if input_node in graph.predecessors[node]:
                bitwidth = input_bitwidth
            else:
                bitwidth = 1
            implementation = graph.nodes[node].implementation
            if implementation in reg:
                reg = reg.add(
                    implementation,
                    reg[implementation].with_attributes(
                        reg[implementation].attributes
                        | ir.attribute(input_bitwidth=bitwidth)
                    ),
                )
        return graph, reg

    return rule
