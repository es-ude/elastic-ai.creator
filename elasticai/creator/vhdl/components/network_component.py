import dataclasses

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.vhdl_files import StaticVHDLFile


@dataclasses.dataclass
class SignalsForBufferlessComponent:
    name: str
    data_width: int

    def _logic_vector(self, suffix, width) -> str:
        return f"signal {self.name}_{suffix} : std_logic_vector({width - 1} downto 0);"

    def _std_logic(self, suffix) -> str:
        return f"signal {self.name}_{suffix} : std_logic := '0';"

    def code(self) -> Code:
        code = [self._std_logic(suffix) for suffix in ("enable", "clock")]
        code.extend(
            [
                self._logic_vector(suffix, width)
                for suffix, width in (
                    ("x", self.data_width),
                    ("y", self.data_width),
                )
            ]
        )

        return code


class SignalsForComponentWithBuffer(SignalsForBufferlessComponent):
    def code(self) -> Code:
        code = list(super().code())
        code.append(self._std_logic("done"))
        code.extend(
            [
                self._logic_vector(suffix, width)
                for suffix, width in (
                    ("x_address", self.x_address_width),
                    ("y_address", self.y_address_width),
                )
            ]
        )

        return code


@dataclasses.dataclass
class ComponentInstantiation:
    name: str

    def _generate_port_map_signals(self) -> list[str]:
        return [
            self._generate_port_to_signal_connection(port)
            for port in (
                "enable",
                "clock",
                "x",
                "y",
            )
        ]

    def _generate_port_to_signal_connection(self, port: str) -> str:
        return f"{port} => {self.name}_{port},"

    @staticmethod
    def _remove_comma_from_last_signal(signals: list[str]) -> list[str]:
        signals[-1] = signals[-1][:-1]
        return signals

    def code(self) -> Code:
        name = self.name
        code = [
            f"{name} : entity work.{name}(rtl)",
            "port map(",
        ]
        mapping = self._generate_port_map_signals()
        mapping = self._remove_comma_from_last_signal(mapping)
        code.extend(mapping)
        code.append(");")
        return code


class BufferedComponentInstantiation(ComponentInstantiation):
    def _generate_port_map_signals(self) -> list[str]:
        signals = super()._generate_port_map_signals()
        signals.extend(
            self._generate_port_to_signal_connection(port)
            for port in ("x_address", "y_address", "done")
        )
        return signals


class NetworkVHDLFile(StaticVHDLFile):
    def __init__(self):
        super().__init__(
            template_package="elasticai.creator.vhdl.templates",
            file_name="network.tpl.vhd",
        )
