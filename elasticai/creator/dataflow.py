from typing import NamedTuple, Union, Any

from elasticai.creator.protocols import Indices, TensorMapping


class DataSource:
    def __init__(self, node: TensorMapping, selection: Union[int, slice, Indices]):
        self.source = node
        self.selection = selection

    @staticmethod
    def _subselection_of(left: Indices, right: Indices):
        def _equal_or_part_of(left: Union[int, slice], right: Union[int, slice]):
            if isinstance(left, int) and isinstance(right, slice):
                return right.start <= left <= right.stop
            if isinstance(left, int) and isinstance(right, int):
                return left == right

        return _equal_or_part_of(left[1], right[1])

    def subsource_of(self, other: "DataSource") -> bool:
        if self._subselection_of(self.selection, other.selection):
            return True
        else:
            return False

    def __eq__(self, other: Any) -> bool:
        if hasattr(other, "source") and hasattr(other, "selection"):
            return other.source == self.source and other.selection == self.selection
        return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"source={self.source.__repr__()}, "
            f"selection={self.selection.__repr__()})"
        )


class DataSink(NamedTuple):
    sources: list[DataSource]
    shape: tuple[int, ...]
    node : Union[TensorMapping,None]
    selection: Union[int, slice, Indices]


DataFlowSpecification = tuple[DataSink, ...]


def sinks_have_common_source(first: DataSink, second: DataSink) -> bool:
    if len(first.sources) != len(second.sources):
        return False
    for first_source in first.sources:
        if not any(x==first_source for x in second.sources):
            return False
    return True


def group_dependent_sinks(sinks: tuple[DataSink, ...]) -> tuple[tuple[DataSink]]:
    groups = []
    sinks = list(sinks)
    while sinks:
        subgroup = [sinks[0]]
        current_sink = sinks[0]
        for sink in sinks:
            if (sinks_have_common_source(current_sink, sink) and sink is not current_sink):
                subgroup.append(sink)
                sinks.remove(sink)
        groups.append(tuple(subgroup))
        sinks.remove(current_sink)
    return tuple(groups)


def represent_grouped_DataFlowSpecification(dataflowspecification:DataFlowSpecification)-> str:
    grouped_dataflow = group_dependent_sinks(dataflowspecification)
    representation = "" 
    for group in grouped_dataflow:
        representation += f"{repr(group[0].sources)} ->"
        for sink in group:
            representation +=f" ({sink.node.__repr__()}, selection = {sink.selection.__repr__()})" 
            if sink != group[-1]:
                representation+=","
        representation += "\n"
    return representation
