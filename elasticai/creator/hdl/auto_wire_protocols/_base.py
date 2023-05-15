def _base_signal_declarations() -> list[str]:
    return [
        "enable : in std_logic",
        "clock : in std_logic",
        "x   : in std_logic_vector($x_width-1 downto 0)",
        "y  : out std_logic_vector($y_width-1 downto 0)",
    ]
