from itertools import chain


def flatten_params(params: list[list[int]]) -> list[int]:
    return list(chain(*params))
