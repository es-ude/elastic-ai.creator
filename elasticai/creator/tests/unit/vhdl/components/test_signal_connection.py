import unittest


class SignalConnector:
    pass


class Node:
    def __init__(self, children: list["Node"]):
        self.children = children
        self.connection_complete = False


class SignalInterface:
    pass


class SignalConnectionTest(unittest.TestCase):

    """
      - node connects its required inputs to provided outputs of parent nodes

      Basic Idea:
        Starting with leaf nodes, connect to higher nodes until no unconnected signals remain.
        Will probably boil down to visitor pattern

    Tests:
      - inserting graph with two nodes and same signals
    """

    def test_graph_with_two_nodes_and_identical_signals_connects_all(self):
        connector = SignalConnector()
        signals = SignalInterface()
        nodes = Node([Node(children=[], signals=signals)], signals=signals)
        connector.connect(nodes)


if __name__ == "__main__":
    unittest.main()
