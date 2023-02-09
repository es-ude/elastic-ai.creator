class DataFlowNode:
    def __init__(self, sinks: set, sources: set):
        pass

    def append(self, child: "DataFlowNode") -> None:
        pass

    def is_satisfied(self) -> bool:
        return True
