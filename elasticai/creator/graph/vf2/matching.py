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

    def compute_candidate_pairs(
        pattern_state: State[TP, T],
        graph_state: State[T, TP],
        depth: int,
    ) -> Iterator[tuple[TP, T]]:
        candidate_pairs: set[tuple[TP, T]] = set()
        out_of_g: set[T] = set(graph_state.iter_out_nodes())
        out_of_p: set[TP] = set(pattern_state.iter_out_nodes())
        if len(out_of_g) == 0 or len(out_of_p) == 0:
            into_g: set[T] = set(graph_state.iter_in_nodes())
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
                    yield pn, gn
        else:
            yield from candidate_pairs

    def is_feasible(pn: TP, gn: T) -> bool:
        partial_pattern_pred = pattern_state.partial_predecessors(pn)
        partial_pattern_succ = pattern_state.partial_successors(pn)
        partial_graph_pred = graph_state.partial_predecessors(gn)
        partial_graph_succ = graph_state.partial_successors(gn)

        pattern_in_node_pred = pattern_state.in_node_predecessors(pn)
        pattern_out_node_pred = pattern_state.out_node_predecessors(pn)
        pattern_in_node_succ = pattern_state.in_node_successors(pn)
        pattern_out_node_succ = pattern_state.out_node_successors(pn)
        pattern_unseen_pred = pattern_state.unseen_predecessors(pn)
        pattern_unseen_succ = pattern_state.unseen_successors(pn)

        full_graph_pred = graph.predecessors[gn]
        full_graph_succ = graph.successors[gn]
        full_pattern_pred = pattern.predecessors[pn]
        full_pattern_succ = pattern.successors[pn]

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
            node in our partial graph match (graph_state.core)
            and vice versa.
            This also prevents a node from being matched if including it would include new edges in the match, that are not present in the pattern, e.g.,
            consider the graph

            a: {b, c}
            b: {c}

            would not match the pattern
            a: {b}
            b: {c}

            because including c would introduce the new edge (a, c) which is not present in the pattern.
            """
            for ps in partial_pattern_pred:
                found = False
                for gs in full_graph_pred:
                    if was_matched(ps, gs):
                        found = True
                        break
                if not found:
                    return False

            for gs in partial_graph_pred:
                found = False
                for ps in full_pattern_pred:
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
            for gs in partial_graph_succ:
                found = False
                for ps in full_pattern_succ:
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

        _in_check = in_check()

        _out_check = out_check()

        _predecessor_check = predecessors_check()

        _successors_check = successors_check()

        _new_check = new_check()

        _constraint = node_constraint(pn, gn)

        result = (
            _in_check
            and _out_check
            and _predecessor_check
            and _successors_check
            and _new_check
            and _constraint
        )

        return result

    def do_match(depth: int) -> Iterator[dict[TP, T]]:
        if pattern_state.is_complete():
            yield dict(pattern_state.iter_matched_pairs())
            return
        pattern_state.next_depth()
        graph_state.next_depth()
        candidate_pairs = list(
            compute_candidate_pairs(pattern_state, graph_state, depth=depth)
        )

        order = {pn: i for i, (pn, _) in enumerate(candidate_pairs)}
        for pn, gn in candidate_pairs:
            if any(n in pattern.nodes and order[n] < order[pn] for n in order):
                continue
            if is_feasible(pn, gn):
                pattern_state.add_pair(pn, gn)
                graph_state.add_pair(gn, pn)
                yield from do_match(depth=depth + 1)
                pattern_state.remove_pair(pn, gn)
                graph_state.remove_pair(gn, pn)

        pattern_state.restore()
        graph_state.restore()

    yield from do_match(depth=0)
