from elasticai.creator.hdl_ir import (
    IrFactory,
    collect_transitive_implementation_closure,
)
from elasticai.creator.ir import Registry

factory = IrFactory()


def test_can_remove_dead_implementations_from_reg():
    g = factory.graph()
    reg = Registry(
        main=g.add_nodes(
            factory.node("a", implementation="a_impl"),
            factory.node("b", implementation="b_impl"),
        ),
        a_impl=g,
        c_impl=g,
    )
    result = collect_transitive_implementation_closure("main", reg)
    assert "a_impl" in result
    assert "c_impl" not in result
