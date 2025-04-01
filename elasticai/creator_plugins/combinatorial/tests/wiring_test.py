from elasticai.creator_plugins.combinatorial.wiring import connect_data_signals


def test_connecting_explicit_data_signals():
    edge = ("src", "dst", ((1, 1),))
    code = tuple(connect_data_signals((edge,)))
    assert code == ("d_in_dst(1) <= d_out_src(1);",)


def test_connecting_data_signals_by_range():
    edge = ("src", "dst", ("range(0, 12)", "range(1, 13)"))
    code = tuple(connect_data_signals((edge,)))
    assert code == ("d_in_dst(12 downto 1) <= d_out_src(11 downto 0);",)
