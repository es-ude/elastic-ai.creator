from elasticai.creator.testing.ghdl_report_parsing import parse_report


def test_parse_ghdl_simulation_results_one_liner():
    simulation_output = (
        "my_test_bench.vhd:64:17:@4ps:(report note):my report message\nsimulation"
        " finished\n"
    )
    expected = [
        {
            "source": "my_test_bench.vhd",
            "line": 64,
            "column": 17,
            "time": "4ps",
            "type": "report note",
            "content": "my report message",
        }
    ]
    assert expected == parse_report(simulation_output)


def test_parse_ghdl_another_line():
    simulation_output = "A:1:2:@B:(C):D\nsimulation finished\n"
    expected = [
        {
            "source": "A",
            "line": 1,
            "column": 2,
            "time": "B",
            "type": "C",
            "content": "D",
        }
    ]
    assert expected == parse_report(simulation_output)


def test_parse_two_lines():
    simulation_output = (
        "source:1:1:@time:(type):content\nsource:1:1:@time:(type):content\nignored"
        " last line\n"
    )
    expected = [
        {
            "source": "source",
            "line": 1,
            "column": 1,
            "time": "time",
            "type": "type",
            "content": "content",
        }
    ] * 2
    assert expected == parse_report(simulation_output)


def test_put_colon_content_in_content_field():
    simulation_output = "A:1:2:@B:(C):D:e:f\n\n"
    assert "D:e:f" == parse_report(simulation_output)[0]["content"]
