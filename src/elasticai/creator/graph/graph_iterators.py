from collections.abc import Hashable, Iterable, Iterator
from typing import Protocol, TypeVar

HashableT = TypeVar("HashableT", bound=Hashable)


class NodeNeighbourFn(Protocol[HashableT]):
    def __call__(self, node: HashableT) -> Iterable[HashableT]: ...


def dfs_iter(successors: NodeNeighbourFn, start: HashableT) -> Iterator[HashableT]:
    visited: set[HashableT] = set()

    def visit(nodes: tuple[HashableT, ...]):
        for n in nodes:
            if n not in visited:
                yield n
                visited.add(n)
                yield from visit(tuple(successors(n)))

    yield from visit((start,))


def bfs_iter_down(
    successors: NodeNeighbourFn, predecessors: NodeNeighbourFn, start: HashableT
) -> Iterator[HashableT]:
    visited: set[HashableT] = set()
    visit_next = sorted(list(successors(start)))
    while len(visit_next) > 0:
        current = visit_next.pop(0)
        if current not in visited:
            visited.add(current)
            yield current
            for child in successors(current):
                if set(predecessors(child)).issubset(visited):
                    visit_next.append(child)


def bfs_iter_up(
    predecessors: NodeNeighbourFn[HashableT],
    successors: NodeNeighbourFn[HashableT],
    start: HashableT,
) -> Iterator[HashableT]:
    return bfs_iter_down(predecessors, successors, start)
