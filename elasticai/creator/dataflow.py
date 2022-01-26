from typing import NamedTuple, Union, Any

from elasticai.creator.protocols import Indices, TensorMapping


class DataSource:
    def __init__(self, source: TensorMapping, selection: Union[int, slice, Indices]):
        self.source = source
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


DataFlowSpecification = tuple[DataSink, ...]


def sinks_have_common_source(first: DataSink, second: DataSink) -> bool:
    pass


def group_dependent_sinks(sinks: tuple[DataSink, ...]) -> tuple[tuple[DataSink]]:
    pass
