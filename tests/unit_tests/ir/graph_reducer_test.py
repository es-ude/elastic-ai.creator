from elasticai.creator import ir

_ir_factory = ir.DefaultIrFactory()


def prepare_executor():
    executor = ir.ExecutionOrderGraphReducer()

    @executor.register()
    def input(node: ir.Node, acc: str):
        return f"{acc}input"

    @executor.register()
    def A(node: ir.Node, acc: str):
        return f"{acc}-tA_{node.name}"

    @executor.register()
    def B(node: ir.Node, acc: str):
        return f"{acc}-tB_{node.name}"

    return executor


def test_gracefully_handle_empty_graph():
    g = _ir_factory.graph()
    executor = prepare_executor()
    assert "" == executor(g, "")


def test_branching_node_graph():
    g = (
        _ir_factory.graph()
        .add_nodes(
            ("input", ir.attribute(type="input")),
            ("h", ir.attribute(type="A")),
            ("e", ir.attribute(type="B")),
            ("l", ir.attribute(type="A")),
            ("o", ir.attribute(type="B")),
        )
        .add_edges(
            ("input", "h"),
            ("h", "e"),
            ("h", "l"),
            ("l", "o"),
            ("e", "o"),
        )
    )
    executor = prepare_executor()
    assert "input-tA_h-tB_e-tA_l-tB_o" == executor(g, "")
