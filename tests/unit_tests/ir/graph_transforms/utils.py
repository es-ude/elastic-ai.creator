from elasticai.creator import ir


def build_graph_from_dict(d: dict[tuple[str, str], list[str]]) -> ir.Implementation:
    g = ir.Implementation()
    for (node, node_type), successors in d.items():
        g.add_node(ir.node(node, node_type))
        for s in successors:
            g.add_edge(ir.edge(node, s))

    return g


def equal_type(a: ir.Node, b: ir.Node) -> bool:
    return a.type == b.type


def find_matches(
    graph: ir.Implementation, pattern: ir.Implementation
) -> list[dict[str, str]]:
    return ir.find_subgraphs(graph, pattern, equal_type)
