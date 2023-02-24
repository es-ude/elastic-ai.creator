def create_instance(
    *,
    name: str,
    entity: str,
    signal_mapping: dict[str, str],
    library: str,
    architecture: str = "rtl",
):
    result = [f"{name} : entity {library}.{entity}({architecture})", "port map(\n"]
    for _from in tuple(signal_mapping.keys())[:-1]:
        _to = signal_mapping[_from]
        result.append(f"\t{_from} => {_to},")
    last_from = tuple(signal_mapping.keys())[-1]
    last_to = signal_mapping[last_from]
    result.append(f"\t{last_from} => {last_to}")
    result.append(");")
    return "\n".join(result)


def signal_definition(
    *,
    name: str,
    width: int,
):
    def vector_signal(name: str, width) -> str:
        return (
            f"signal {name} : std_logic_vector({width - 1} downto 0) := ('other' =>"
            " '0');"
        )

    def logic_signal(name: str) -> str:
        return f"signal {name} : std_logic := '0';"

    if width > 0:
        return vector_signal(name, width)
    else:
        return logic_signal(name)
