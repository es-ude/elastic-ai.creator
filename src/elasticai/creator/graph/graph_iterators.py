from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Protocol, Self


class _NodeLT(Hashable, Protocol):
    def __lt__(self, other: Self, /) -> bool: ...


class _NodeGT(Hashable, Protocol):
    def __gt__(self, other: Self, /) -> bool: ...


type _Node = _NodeGT | _NodeLT

type NodeNeighbourFn[T: _Node] = Callable[[T], Iterable[T]]


def dfs_iter[T: _Node](successors: NodeNeighbourFn[T], start: T) -> Iterator[T]:
    visited: set[T] = set()

    def visit(nodes: tuple[T, ...]) -> Iterator[T]:
        for n in nodes:
            if n not in visited:
                yield n
                visited.add(n)
                yield from visit(tuple(successors(n)))

    yield from visit((start,))


def bfs_iter_down[T: _Node](
    successors: NodeNeighbourFn[T],
    predecessors: NodeNeighbourFn[T],
    start: T | set[T],
) -> Iterator[T]:
    """Iterate graph nodes in breadth first.

    This ensures that no node will be visited before
    each of its predecessors was visited.
    """
    visited: set[T] = set()
    if isinstance(start, set):
        visit_next: list[T] = []
        for n in start:
            for succ in successors(n):
                visit_next.append(succ)
        visit_next = list(sorted(set(visit_next)))
    else:
        visit_next = sorted(list(successors(start)))
    while len(visit_next) > 0:
        current = visit_next.pop(0)
        if current not in visited:
            visited.add(current)
            yield current
            for child in successors(current):
                if set(predecessors(child)).issubset(visited):
                    visit_next.append(child)


def bfs_iter_up[T: _Node](
    predecessors: NodeNeighbourFn[T],
    successors: NodeNeighbourFn[T],
    start: T | set[T],
) -> Iterator[T]:
    """Iterate graph nodes in breadth first.

    This ensures that no node will be visited before
    each of its successors was visited.
    """
    # we ignore type because type checker fails to get that set is not hashable
    return bfs_iter_down(successors=predecessors, predecessors=successors, start=start)
