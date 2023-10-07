def instantiate_lut(
    lut_name: str,
    instance_index: int,
    input_slice: tuple[int, int],
    output_slice: tuple[int, int],
) -> str:
    return f"""i_{lut_name}_{instance_index} : entity work.{lut_name}(rtl)
port map(
    clock => clock,
    enable => enable,
    x => x({input_slice[1]}-1 downto {input_slice[0]}),
    y => y({output_slice[1]}-1 downto {output_slice[0]})
);"""
