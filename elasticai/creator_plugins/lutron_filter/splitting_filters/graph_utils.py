import networkx as nx
from lutron.ir.utils.filter_parameters import FilterParameters
from lutron.ir.utils.index_generators import GroupedFilterIndexGenerator


def build_graph_for_filter_channels(
    filters: tuple[FilterParameters, ...],
) -> nx.MultiDiGraph:
    g = nx.DiGraph()

    def create_idx_generator(channels, groups):
        return GroupedFilterIndexGenerator(
            FilterParameters(
                kernel_size=1,
                in_channels=channels,
                groups=groups,
                out_channels=channels,
            )
        )

    def add_edges(id, filter):
        from_id = id
        to_id = id + 1
        from_gen = create_idx_generator(filter.in_channels, filter.groups)
        to_gen = create_idx_generator(filter.out_channels, filter.groups)
        for step in from_gen.steps():
            for step2 in to_gen.steps():
                for group1, group2 in zip(step.groups(), step2.groups()):
                    for idx0 in group1:
                        for idx1 in group2:
                            g.add_edge((from_id, idx0), (to_id, idx1))

    for idx, filter in enumerate(filters):
        add_edges(idx, filter)

    return g


def draw_filter_graph(g):
    subsets = {}
    for n in g.nodes:
        subset = subsets.get(n[0], list())
        subset.append(n)
        subsets[n[0]] = subset
    layout = nx.drawing.multipartite_layout(G=g, subset_key=subsets)
    nx.draw(g, pos=layout)


def cross_layer_information_flow(g) -> float:
    """A measure for how many of the input channels can be reached by tracing back the data flow from a single output
    channel on average. 1.0 meaning every input channel can influence the result of every output channel.
    """
    last_layer_id = 0
    for n in g.nodes:
        if n[0] > last_layer_id:
            last_layer_id = n[0]
    ancestors = {}
    for node in g.nodes:
        if node[0] == last_layer_id:
            ancestors[node] = set()
            for pre in g.predecessors(node):
                for prepre in g.predecessors(pre):
                    ancestors[node].add(prepre)
    avg_num_ancestors = 0
    for v in ancestors.values():
        avg_num_ancestors += len(v)
    avg_num_ancestors /= len(ancestors)
    num_in_channels = 0
    for n in g.nodes:
        if n[0] == 0:
            num_in_channels += 1
    cross_channel_information_flow = avg_num_ancestors / num_in_channels
    return cross_channel_information_flow


def cli(f):
    g = build_graph_for_filter_channels(f)
    return cross_layer_information_flow(g)
