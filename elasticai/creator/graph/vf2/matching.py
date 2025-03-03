from collections.abc import Iterator
from typing import TypeVar

from elasticai.creator.graph.graph import Graph

from .._types import NodeConstraintFn
from .state import State

T = TypeVar("T")
TP = TypeVar("TP")


class MatchError(Exception):
    def __init__(self):
        super().__init__("No match found")


def match(
    pattern: Graph[TP],
    graph: Graph[T],
    node_constraint: NodeConstraintFn[TP, T] = lambda _, __: True,
) -> dict[TP, T]:
    try:
        return next(_match(pattern, graph, node_constraint))
    except StopIteration:
        raise MatchError()


def find_all_matches(
    pattern: Graph[TP],
    graph: Graph[T],
    node_constraint: NodeConstraintFn[TP, T] = lambda _, __: True,
) -> list[dict[TP, T]]:
    return list(_match(pattern, graph, node_constraint))


def _match(
    pattern: Graph[TP],
    graph: Graph[T],
    node_constraint: NodeConstraintFn[TP, T] = lambda _, __: True,
) -> Iterator[dict[TP, T]]:
    """
    Implementation of the VF2 algorithm by Cordella et al. 2004
    """
    pattern_state: State[TP, T] = State(
        graph=pattern,
    )
    graph_state: State[T, TP] = State(
        graph=graph,
    )

    def get_all_matches() -> Iterator[dict[TP, T]]:
        nonlocal pattern_state, graph_state
        matched_graph_nodes: set[T] = set()
        while True:
            do_match(exclude_nodes=matched_graph_nodes, depth=0)
            if pattern_state.is_complete():
                last_match = dict(pattern_state.iter_matched_pairs())
                matched_graph_nodes.update(last_match.values())
                yield last_match
                pattern_state = State(graph=pattern)
                graph_state = State(graph=graph)
            else:
                break

    def compute_candidate_pairs(
        pattern_state: State[TP, T],
        graph_state: State[T, TP],
        excluded_nodes: set[T],
        depth: int,
    ) -> Iterator[tuple[TP, T]]:
        candidate_pairs: set[tuple[TP, T]] = set()
        out_of_g: set[T] = set(graph_state.iter_out_nodes()).difference(excluded_nodes)
        out_of_p: set[TP] = set(pattern_state.iter_out_nodes())
        if len(out_of_g) == 0 or len(out_of_p) == 0:
            into_g: set[T] = set(graph_state.iter_in_nodes()).difference(excluded_nodes)
            into_p: set[TP] = set(pattern_state.iter_in_nodes())
            for pn in into_p:
                for gn in into_g:
                    candidate_pairs.add((pn, gn))

        else:
            for pn in out_of_p:
                for gn in out_of_g:
                    candidate_pairs.add((pn, gn))

        if len(candidate_pairs) == 0 and depth == 0:
            for pn in pattern.nodes:
                for gn in graph.nodes:
                    if gn not in excluded_nodes:
                        yield pn, gn
        else:
            yield from candidate_pairs

    def is_feasible(pn: TP, gn: T) -> bool:
        partial_pattern_pred = pattern_state.partial_predecessors(pn)
        partial_pattern_succ = pattern_state.partial_successors(pn)
        pattern_in_node_pred = pattern_state.in_node_predecessors(pn)
        pattern_out_node_pred = pattern_state.out_node_predecessors(pn)
        pattern_in_node_succ = pattern_state.in_node_successors(pn)
        pattern_out_node_succ = pattern_state.out_node_successors(pn)
        pattern_unseen_pred = pattern_state.unseen_predecessors(pn)
        pattern_unseen_succ = pattern_state.unseen_successors(pn)

        full_graph_pred = graph.predecessors[gn]
        full_graph_succ = graph.successors[gn]

        graph_in_node_pred = graph_state.in_node_predecessors(gn)
        graph_out_node_pred = graph_state.out_node_predecessors(gn)

        graph_in_node_succ = graph_state.in_node_successors(gn)
        graph_out_node_succ = graph_state.out_node_successors(gn)

        graph_unseen_pred = graph_state.unseen_predecessors(gn)
        graph_unseen_succ = graph_state.unseen_successors(gn)

        def was_matched(ps, gs):
            return pattern_state.contains_pair(ps, gs)

        def predecessors_check():
            """
            ensure that each predecessors of the considered nodes
            that is in our partial pattern match (pattern_state.core) also has a corresponding
            node in our partial graph match (graph_state.core).
            The paper states we'd have to match vice versa, but that would prevent actual subgraph matching.
            """
            for ps in partial_pattern_pred:
                found = False
                for gs in full_graph_pred:
                    if was_matched(ps, gs):
                        found = True
                        break
                if not found:
                    return False

            return True

        def successors_check():
            """
            analogue to predecessor check
            """
            for ps in partial_pattern_succ:
                found = False
                for gs in full_graph_succ:
                    if was_matched(ps, gs):
                        found = True
                        break
                if not found:
                    return False

            return True

        def in_check():
            "checks nodes that are might be added"
            return len(graph_in_node_pred) >= len(pattern_in_node_pred) and len(
                graph_in_node_succ
            ) >= len(pattern_in_node_succ)

        def out_check():
            return len(graph_out_node_pred) >= len(pattern_out_node_pred) and len(
                graph_out_node_succ
            ) >= len(pattern_out_node_succ)

        def new_check():
            """Considers nodes that might be added in current_depth + 2"""
            return len(graph_unseen_pred) >= len(pattern_unseen_pred) and len(
                graph_unseen_succ
            ) >= len(pattern_unseen_succ)

        result = (
            in_check()
            and out_check()
            and predecessors_check()
            and successors_check()
            and new_check()
            and node_constraint(pn, gn)
        )

        return result

    result: dict[TP, T] = {}

    def do_match(exclude_nodes: set[T], depth: int) -> None:
        nonlocal result
        if pattern_state.is_complete():
            return
        pattern_state.next_depth()
        graph_state.next_depth()

        for pn, gn in compute_candidate_pairs(
            pattern_state, graph_state, excluded_nodes=exclude_nodes, depth=depth
        ):
            if pattern_state.contains_node_with_lower_id(pn):
                continue
            if is_feasible(pn, gn):
                pattern_state.add_pair(pn, gn)
                graph_state.add_pair(gn, pn)
                do_match(exclude_nodes=exclude_nodes, depth=depth + 1)
                if pattern_state.is_complete():
                    return
                pattern_state.remove_pair(pn, gn)
                graph_state.remove_pair(gn, pn)
        pattern_state.restore()
        graph_state.restore()

    return get_all_matches()
