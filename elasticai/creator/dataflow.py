from typing import NamedTuple, Callable

from elasticai.creator.protocols import Tensor, Index


class DataSource(NamedTuple):
    source: Callable[[...], Tensor]
    selection: Index


class DataSink(NamedTuple):
    sources: list[DataSource]
    shape: tuple[int, ...]


DataFlowSpecification = tuple[DataSink, ...]


def sinks_have_common_source(first: DataSink, second: DataSink) -> bool:
    for source in first.sources:
        if source in second.sources and len(source.selection) > 0:
            return True
    return False


def group_dependent_sinks(sinks: tuple[DataSink, ...]) -> tuple[tuple[DataSink]]:
    pass
