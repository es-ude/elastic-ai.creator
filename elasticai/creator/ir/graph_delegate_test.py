from .graph_delegate import GraphDelegate
from .graph_iterators import bfs_iter_up, dfs_pre_order


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

    actual = tuple(bfs_iter_up(g.get_predecessors, "5"))
    assert set(actual[0:2]) == {"3", "6"}
    assert (set(actual[2:4]) == {"1", "2"} and actual[4] == "4") or (
        set(actual[3:5]) == {"1", "2"} and actual[2] == "4"
    )


def test_iterating_depth_first_preorder():
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

    actual = tuple(dfs_pre_order(g.get_successors, "0"))
    print(tuple(g.get_successors("0")))
    expected = ("0", "1", "3", "5", "2", "4", "6")
    assert actual == expected
