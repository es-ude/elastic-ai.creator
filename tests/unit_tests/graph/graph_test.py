import pytest

from elasticai.creator.graph import BaseGraph as GraphDelegate
from elasticai.creator.graph import bfs_iter_up, dfs_iter


def test_yield_node_in_case_it_has_no_successors():
    g = GraphDelegate.from_dict(
        {
            "0": [],
        }
    )
    actual = tuple(g.iter_nodes())
    assert actual == ("0",)


def test_iterating_breadth_first_upwards():
    g = GraphDelegate()
    """
             0
             |
          /-----\
          |     |
          1     2
          | /---+
          |/    |
          3     4
          |     |
          |     6
          |/----+
          5
    """
    g = GraphDelegate.from_dict(
        {
            "0": ["1", "2"],
            "1": ["3"],
            "2": ["3", "4"],
            "3": ["5"],
            "4": ["6"],
            "6": ["5"],
        }
    )

    actual = tuple(bfs_iter_up(g.get_predecessors, g.get_successors, "5"))
    assert actual == ("3", "6", "1", "4", "2", "0")


@pytest.mark.parametrize(
    ["adjacencies", "expected"],
    [
        (
            #      0
            #      |
            #   /-----\
            #   |     |
            #   1     2
            #   | /---+
            #   |/    |
            #   3     4
            #   |     |
            #   |     6
            #   |/----+
            #   5
            {
                "0": ["1", "2"],
                "1": ["3"],
                "2": ["3", "4"],
                "3": ["5"],
                "4": ["6"],
                "6": ["5"],
            },
            {
                ("0", "2", "4", "1", "3", "5", "6"),
                ("0", "2", "3", "5", "4", "6", "1"),
                ("0", "1", "3", "5", "2", "4", "6"),
                ("0", "2", "4", "6", "5", "1", "3"),
                ("0", "2", "4", "6", "5", "3", "1"),
            },
        ),
        (
            {
                "0": ["1", "4"],
                "1": ["2"],
                "2": ["3"],
                "3": [],
                "4": ["5", "6"],
                "5": [],
                "6": [],
            },
            {
                ("0", "4", "5", "6", "1", "2", "3"),
                ("0", "4", "6", "5", "1", "2", "3"),
                ("0", "1", "2", "3", "4", "5", "6"),
                ("0", "1", "2", "3", "4", "6", "5"),
            },
        ),
        (
            {"0": ["1", "2"], "1": [], "2": []},
            {
                ("0", "1", "2"),
                ("0", "2", "1"),
            },
        ),
    ],
)
def test_iterate_dfs(adjacencies, expected):
    g = GraphDelegate.from_dict(adjacencies)

    actual = tuple(dfs_iter(g.get_successors, "0"))
    assert actual in expected
