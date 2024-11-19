from collections.abc import Callable, Iterable, Iterator
from typing import Hashable, TypeAlias, TypeVar

HashableT = TypeVar("HashableT", bound=Hashable)

NodeNeighbourFn: TypeAlias = Callable[[HashableT], Iterable[HashableT]]


def dfs_pre_order(successors: NodeNeighbourFn, start: HashableT) -> Iterator[HashableT]:
    visited: set[HashableT] = set()

    def visit(nodes: tuple[HashableT, ...]):
        for n in nodes:
            if n not in visited:
                yield n
                visited.add(n)
                yield from visit(tuple(successors(n)))

    yield from visit((start,))


def bfs_iter_down(successors: NodeNeighbourFn, start: HashableT) -> Iterator[HashableT]:
    visited: set[HashableT] = set()
    visit_next = [start]
    while len(visit_next) > 0:
        current = visit_next.pop(0)
        for p in successors(current):
            if p not in visited:
                yield p
                visited.add(p)
                visit_next.append(p)
        visited.add(current)


def bfs_iter_up(predecessors: NodeNeighbourFn, start: HashableT) -> Iterator[HashableT]:
    return bfs_iter_down(predecessors, start)
