from .iterator_utils import batched


def structure_flat_CxN(vector, channels):
    """Assumes a vector of the following format
    (x[1, 1], ..., x[C, 1], x[1, 2], ..., x[C, 2], ..., x[1, N], ..., x[C, N])
    and reshapes it to a tensor y, such that
    y[i] contains all channels of step i
    """
    return tuple(batched(vector, channels))


def reshape_flat_CxN_groupwise(vector, channels, groups):
    """Assumes a vector of the following format
    (x[1, 1], ..., x[C, 1], x[1, 2], ..., x[C, 2], ..., x[1, N], ..., x[C, N])
    and reshapes it to a tensor y, such that
    y[i] contains a flat vector for all spatial steps and all channels in group i in the
    same CxN format.
    E.g.:
    >>> reshape_flat_CxN_groupwise(tuple(range(12)), channels=4, groups=2)
    ... ((0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11))
    """
    structured = structure_flat_CxN(vector, channels)
    group_size = channels // groups
    grouped_spatial_steps = ((batched(x, group_size)) for x in structured)

    groups_combined_across_spatial_steps = zip(*grouped_spatial_steps)
    flattened_groups = tuple(
        tuple(y for x in xs for y in x) for xs in groups_combined_across_spatial_steps
    )
    return flattened_groups
