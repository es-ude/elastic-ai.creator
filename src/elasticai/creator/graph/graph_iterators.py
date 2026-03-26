from collections.abc import Hashable, Iterable, Iterator
from typing import Protocol, TypeVar, overload

HashableT = TypeVar("HashableT", bound=Hashable)


class NodeNeighbourFn(Protocol[HashableT]):
    def __call__(self, node: HashableT, /) -> Iterable[HashableT]: ...


def dfs_iter(successors: NodeNeighbourFn, start: HashableT) -> Iterator[HashableT]:
    visited: set[HashableT] = set()

    def visit(nodes: tuple[HashableT, ...]):
        for n in nodes:
            if n not in visited:
                yield n
                visited.add(n)
                yield from visit(tuple(successors(n)))

    yield from visit((start,))


@overload
def bfs_iter_down[T: Hashable](
    successors: NodeNeighbourFn, predecessors: NodeNeighbourFn, start: set[T]
) -> Iterator[T]: ...


@overload
def bfs_iter_down[T: Hashable](
    successors: NodeNeighbourFn, predecessors: NodeNeighbourFn, start: T
) -> Iterator[T]: ...


def bfs_iter_down[T](
    successors: NodeNeighbourFn,
    predecessors: NodeNeighbourFn,
    start: T | set[T],
) -> Iterator[T]:
    """Iterate graph nodes in breadth first.

    This ensures that no node will be visited before
    each of its predecessors was visited.
    """
    visited: set[T] = set()
    if isinstance(start, set):
        visit_next = []
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


@overload
def bfs_iter_up[T: Hashable](
    successors: NodeNeighbourFn, predecessors: NodeNeighbourFn, start: T, /
) -> Iterator[T]: ...


@overload
def bfs_iter_up[T: Hashable](
    successors: NodeNeighbourFn, predecessors: NodeNeighbourFn, start: set[T], /
) -> Iterator[T]: ...


def bfs_iter_up[T](
    predecessors: NodeNeighbourFn[T],
    successors: NodeNeighbourFn[T],
    start: T | set[T],
) -> Iterator[T]:
    """Iterate graph nodes in breadth first.

    This ensures that no node will be visited before
    each of its successors was visited.
    """
    # we ignore type because type checker fails to get that set is not hashable
    return bfs_iter_down(successors=predecessors, predecessors=successors, start=start)  # type: ignore
