from typing import Iterator

from elasticai.creator.hdl_ir import Registry


def m_to_n_lut_cost(m: int, n: int) -> int:
    return int(n / 3.0 * (2 ** (m - 4) - (-1) ** m))


def estimated_lower_lut_cost_by_implementation(
    reg: Registry,
) -> Iterator[tuple[str, int]]:
    for name, impl in reg.items():
        if impl.type == "lutron":
            m, n = (impl.attributes["input_size"], impl.attributes["output_size"])
            yield name, m_to_n_lut_cost(m, n)


def estimate_lut_cost_lower_bound(reg: Registry) -> int:
    io_sizes: list[tuple[int, int]] = []
    for impl in reg.values():
        if impl.type == "lutron":
            io_sizes.append(
                (impl.attributes["input_size"], impl.attributes["output_size"])
            )
    total = 0
    for m, n in io_sizes:
        total += m_to_n_lut_cost(m, n)
    return total
