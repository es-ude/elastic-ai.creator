from .ghdl_report_parsing import parse


def test_parse_ghdl_simulation_results_one_liner():
    simulation_output = "my_test_bench.vhd:64:17:@4ps:(report note):my report message"
    expected = {
        "source": "my_test_bench.vhd",
        "line": 64,
        "column": 17,
        "time": "4ps",
        "type": "report note",
        "content": "my report message",
    }
    assert expected == parse(simulation_output)


def test_parse_ghdl_another_line():
    simulation_output = "A:1:2:@B:(C):D"
    expected = {
        "source": "A",
        "line": 1,
        "column": 2,
        "time": "B",
        "type": "C",
        "content": "D",
    }
    assert expected == parse(simulation_output)
