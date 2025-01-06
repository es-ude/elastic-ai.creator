from typing import Iterator

from lutron.ir import ImplementationRegistry


def m_to_n_lut_cost(m: int, n: int) -> int:
    return int(n / 3.0 * (2 ** (m - 4) - (-1) ** m))


def estimated_lower_lut_cost_by_implementation(
    reg: ImplementationRegistry,
) -> Iterator[tuple[str, int]]:
    for name, impl in reg.items():
        if impl.type == "lutron":
            m, n = (impl.attributes["input_size"], impl.attributes["output_size"])
            yield name, m_to_n_lut_cost(m, n)


def estimate_lut_cost_lower_bound(reg: ImplementationRegistry) -> int:
    io_sizes: list[tuple[int, int]] = []
    for name, impl in reg.items():
        if impl.type == "lutron":
            io_sizes.append(
                (impl.attributes["input_size"], impl.attributes["output_size"])
            )
    total = 0
    for m, n in io_sizes:
        total += m_to_n_lut_cost(m, n)
    return total


def estimate_unrolled_lut_cost_by_implementation(
    reg: ImplementationRegistry,
) -> Iterator[tuple[str, int]]:
    for name, impl in reg.items():
        size = 0
        for node in impl.iter_nodes():
            if node.type == "lutron":
                m, n = node.input_shape[0], node.output_shape[0]
                size += m_to_n_lut_cost(m, n)
        yield name, size


def estimate_unrolled_lut_cost(reg: ImplementationRegistry) -> int:
    io_sizes: list[tuple[int, int]] = []

    for name, impl in reg.items():
        for node in impl.iter_nodes():
            if node.type == "lutron":
                io_sizes.append((node.input_shape[0], node.output_shape[0]))

    total = 0
    for m, n in io_sizes:
        total += m_to_n_lut_cost(m, n)
    return total
